import os
import time
import asyncio
from collections import deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openvino_genai as ov_genai
import uvicorn

app = FastAPI(title="NPU LLM Offloader Server")
npu_lock = asyncio.Lock()
pipe = None
loaded_model_path = None

# --- Single source of truth for the prompt ceiling --------------------------
# MAX_PROMPT_LEN is a LOAD-TIME constant: it sizes compile-time host/NPU
# buffers (KV cache etc.) and is NOT something main.py can change per
# request. Empirically verified on this machine (2026-09-01 probe):
# compiling at 6144 succeeds down to ~1.2GB free physical RAM / 91% memory
# load, so the compile OOM seen in the field happened under total host
# memory (commit) exhaustion where NO reasonable MAX_PROMPT_LEN would have
# compiled — lowering this value does not fix that; freeing RAM does.
# main.py derives its live prompt target from /api/health
# (effective_max_prompt_len), so this constant stays authoritative.
MAX_PROMPT_LEN = 6144

# Actual ceiling of the pipeline that is currently loaded. Set on every
# successful load path; None means "loaded WITHOUT a MAX_PROMPT_LEN
# constraint" (the silent retry path) — an unknown/default ceiling that
# main.py must treat conservatively. Reported via /api/health.
effective_max_prompt_len = None


# --- Stage 1 generation-speed observability (cooldown investigation) -------
# Rolling window of the last 20 REAL generation calls: a throughput proxy
# (chars/sec) for a future cooldown decision. RECORDING ONLY — nothing here
# pauses, delays, or modifies any NPU call. Pure bookkeeping around calls
# that already happen: the append is O(1) and happens AFTER the generation
# completes, outside npu_lock/gpu_lock; the health read snapshots the deque
# (deque append is atomic under the GIL, and appends come only from the
# event loop thread). Resets on restart by design — this is a live-session
# trend signal, not a permanent log.
GEN_WINDOW_SIZE = 20
_gen_history: deque = deque(maxlen=GEN_WINDOW_SIZE)


def _log_free_ram(context: str) -> None:
    """Prints host memory state — the diagnostic that was missing when the
    compile-time ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY happened in the field.
    ctypes/GlobalMemoryStatusEx: no psutil dependency."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        print(f"[npu_server] host RAM at {context}: mem_load={st.dwMemoryLoad}% "
              f"free_phys={st.ullAvailPhys // (1024 * 1024)}MB "
              f"free_commit={st.ullAvailPageFile // (1024 * 1024)}MB")
    except Exception as e:
        print(f"[npu_server] could not read host memory state: {e}")

# --- LoRA adapter (style + formatting fine-tune, see PROJECT_STATE.txt) ---
# Points at the actual .safetensors file produced by the QLoRA training
# run, not just the containing directory — matches the documented
# ov_genai.Adapter() signature (adapter = ov_genai.Adapter("path/to/
# lora_adapter.safetensors")) confirmed during earlier feasibility
# research, rather than assuming a bare directory path works.
ADAPTER_DIR = "./lora_adapter"
ADAPTER_PATH = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
# Blend strength — this is OpenVINO GenAI's FINAL merged scaling factor,
# NOT a direct carryover from the training-time lora_alpha/r ratio (r=16,
# alpha=16 was the training config; 0.75 here is a separate runtime value
# to start from). Sweep 0.5/0.75/1.0 if results are off once the
# acceptance test (known violating prompts + style_check()) has been run —
# don't assume 0.75 is correct without testing.
ADAPTER_ALPHA = 0.4


def _load_adapter_config():
    """
    Loads the trained LoRA adapter if the weights are present at
    ADAPTER_PATH, returning an AdapterConfig ready to attach to any
    LLMPipeline construction. Returns None (adapter disabled, base model
    behavior unchanged) if the file is missing or loading fails for any
    reason — deliberately non-fatal. The adapter weights may not yet be
    downloaded to this machine as of this update (see PROJECT_STATE.txt),
    so the server must still start and serve the base model correctly in
    that case rather than crashing on startup.
    """
    if not os.path.isfile(ADAPTER_PATH):
        print(f"[LoRA] No adapter found at '{ADAPTER_PATH}' — running base model without style adapter.")
        return None
    try:
        adapter = ov_genai.Adapter(ADAPTER_PATH)
        config = ov_genai.AdapterConfig(adapter, ADAPTER_ALPHA)
        print(f"[LoRA] Loaded adapter from '{ADAPTER_PATH}' (alpha={ADAPTER_ALPHA}).")
        return config
    except Exception as e:
        print(f"[LoRA] Failed to load adapter from '{ADAPTER_PATH}': {e} — running base model without style adapter.")
        return None


adapter_config = _load_adapter_config()


def _build_pipeline(path, device, **kwargs):
    """
    Constructs an LLMPipeline, attaching the trained LoRA adapter (style +
    formatting fine-tune) if one loaded successfully at startup.
    Centralized here so every pipeline construction site — the primary
    NPU load and all its fallback branches below, plus the lazily-compiled
    GPU overflow pipeline in get_gpu_pipeline() — picks up the adapter
    consistently. Without this, a GPU-routed oversized prompt would
    silently fall back to the un-adapted base model's behavior while
    NPU-routed requests got the adapter, an inconsistency worth avoiding.
    """
    if adapter_config is not None:
        kwargs = {**kwargs, "adapters": adapter_config}
    return ov_genai.LLMPipeline(path, device, **kwargs)

# --- GPU overflow pipeline (lazy-compiled) ---
# NPU is compiled with a hard MAX_PROMPT_LEN=6144 at startup — any
# request whose real token count exceeds that fails outright (500),
# no truncation attempted, no fallback. Rather than trying to
# hard-cap every possible retrieval source upstream (already done in
# rag.py, but that's defense-in-depth, not a guarantee against every
# future oversized prompt), this adds a genuine fallback: an
# oversized prompt gets routed to a GPU-compiled pipeline instead of
# just failing.
#
# GPU pipeline is NOT compiled at startup — only on first request
# that actually needs it. Compiling it eagerly would mean holding two
# full compiled model copies in memory at all times, and this
# machine's RAM is already known to be tight (~15.4GB, see the
# OllamaThrottle rationale in main.py). Lazy compilation means the
# common case (prompts fitting on NPU) never pays that memory cost at
# all — only the rare overflow case does, once, and then it stays
# warm for the rest of the process lifetime.
#
# gpu_compile_lock guards the lazy-compile step itself so two
# concurrent oversized requests can't both try to compile the GPU
# pipeline at once.
gpu_pipe = None
gpu_compile_failed = False  # sticky flag — don't retry a compile that's already failed once per process lifetime
gpu_compile_lock = asyncio.Lock()
gpu_lock = asyncio.Lock()  # serializes GPU inference, separate from npu_lock so a long
                            # GPU-routed request doesn't block normal-sized NPU requests

# Requests estimated above this many tokens get routed to the GPU
# pipeline instead of NPU. Set with real headroom under NPU's compiled
# MAX_PROMPT_LEN=6144 — not right up against the ceiling, since the
# token estimate (even the real tokenizer count, see estimate_tokens())
# can still be a few tokens off from what the NPU pipeline itself
# ultimately counts internally (chat-template / special-token handling).
NPU_SAFE_PROMPT_TOKENS = 5500

# --- GPU fallback kill switch ---
# This machine has a known Vulkan driver issue. get_gpu_pipeline()'s
# `except Exception` around the GPU compile call can only catch a clean
# Python-level failure — a Vulkan driver fault during compile is a
# plausible native crash that Python cannot catch at all, which would
# take down this entire process with no traceback (matches an incident
# where npu_server.py died outright on an oversized prompt, see
# PROJECT_STATE.txt). Until the Vulkan issue is fixed (or GPU compile is
# isolated into its own subprocess so a native crash there can't kill
# this process), GPU fallback is disabled outright: an oversized prompt
# falls straight to "attempt NPU anyway", which fails cleanly with a
# catchable 500 instead of risking a process-killing crash. Flip this
# back to True only once one of those two things is actually fixed.
GPU_FALLBACK_ENABLED = False

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None

@app.on_event("startup")
def startup_event():
    global pipe, loaded_model_path
    
    # Switched from the 8B Llama 3.1 (Ollama) model to the downloaded 7B
    # Qwen2.5 OpenVINO IR model. Llama 3.1 8B NPU-OV entry removed from
    # this list entirely (was previously first-priority) so it can never
    # get silently picked even if that directory exists/gets downloaded
    # later — Qwen 2.5 7B is now the explicit primary choice, with the
    # 1.5B variant kept only as a last-resort fallback if 7B fails to load.
    models_to_try = [
        ("./qwen2.5-7b-instruct-int4-ov", "Qwen 2.5 7B"),
        ("./qwen2.5-1.5b-instruct-int4-ov", "Qwen 2.5 1.5B")
    ]
    
    selected_path = None
    selected_name = None
    
    for path, name in models_to_try:
        # Check if the folder exists and has a .bin file larger than 100MB (complete download check)
        if os.path.isdir(path):
            try:
                files = os.listdir(path)
                has_bin = any(
                    f.endswith('.bin') and os.path.getsize(os.path.join(path, f)) > 100 * 1024 * 1024
                    for f in files
                )
                has_xml = any(f.endswith('.xml') for f in files)
                if has_bin and has_xml:
                    selected_path = path
                    selected_name = name
                    break
            except Exception:
                continue
            
    if not selected_path:
        print("ERROR: No valid model directory found. Make sure download_model.py completed.")
        return
        
    # Parse device from command line if provided (default to NPU)
    import sys
    device = "NPU"
    for arg in sys.argv:
        if arg.startswith("--device="):
            device = arg.split("=")[1]
        elif arg == "--device" and sys.argv.index(arg) + 1 < len(sys.argv):
            device = sys.argv[sys.argv.index(arg) + 1]
            
    print(f"Loading/compiling {selected_name} from '{selected_path}' onto device '{device}' with MAX_PROMPT_LEN=6144...")
    start_time = time.time()
    global pipe, loaded_model_path, effective_max_prompt_len
    _log_free_ram("startup (before model load)")
    try:
        pipe = _build_pipeline(selected_path, device, MAX_PROMPT_LEN=MAX_PROMPT_LEN)
        loaded_model_path = selected_path
        effective_max_prompt_len = MAX_PROMPT_LEN
        print(f"Model compiled successfully on '{device}' with MAX_PROMPT_LEN={MAX_PROMPT_LEN} in {time.time() - start_time:.2f} seconds!")
    except Exception as e:
        _log_free_ram(f"load failure on '{device}' with MAX_PROMPT_LEN={MAX_PROMPT_LEN}")
        print(f"Failed to load onto '{device}' with MAX_PROMPT_LEN: {e}")
        print("Retrying on same device without MAX_PROMPT_LEN...")
        try:
            pipe = _build_pipeline(selected_path, device)
            loaded_model_path = selected_path
            # effective_max_prompt_len stays None: the ceiling is now
            # OpenVINO's default, not our constant — unknown to us.
            print(f"Model compiled successfully on '{device}' (default MAX_PROMPT_LEN) in {time.time() - start_time:.2f} seconds!")
        except Exception as e2:
            _log_free_ram(f"load failure on '{device}' without MAX_PROMPT_LEN")
            print(f"Failed retry on same device: {e2}")

            # Try fallbacks in order of preference
            fallbacks = ["HETERO:NPU,GPU,CPU", "HETERO:NPU,CPU", "GPU", "CPU"]
            success = False
            for fallback in fallbacks:
                if fallback == device:
                    continue
                print(f"Attempting fallback to '{fallback}' with MAX_PROMPT_LEN={MAX_PROMPT_LEN}...")
                try:
                    start_fallback = time.time()
                    pipe = _build_pipeline(selected_path, fallback, MAX_PROMPT_LEN=MAX_PROMPT_LEN)
                    loaded_model_path = selected_path
                    effective_max_prompt_len = MAX_PROMPT_LEN
                    print(f"Model loaded successfully on '{fallback}' fallback with MAX_PROMPT_LEN={MAX_PROMPT_LEN} in {time.time() - start_fallback:.2f} seconds!")
                    success = True
                    break
                except Exception as fe:
                    _log_free_ram(f"fallback load failure on '{fallback}' with MAX_PROMPT_LEN")
                    print(f"Failed fallback to '{fallback}' with MAX_PROMPT_LEN: {fe}")
                    print(f"Retrying fallback to '{fallback}' without MAX_PROMPT_LEN...")
                    try:
                        start_fallback = time.time()
                        pipe = _build_pipeline(selected_path, fallback)
                        loaded_model_path = selected_path
                        print(f"Model loaded successfully on '{fallback}' fallback (default MAX_PROMPT_LEN) in {time.time() - start_fallback:.2f} seconds!")
                        success = True
                        break
                    except Exception as fe2:
                        _log_free_ram(f"fallback load failure on '{fallback}' without MAX_PROMPT_LEN")
                        print(f"Failed fallback to '{fallback}' completely: {fe2}")

            if not success:
                _log_free_ram("total load failure (all devices/fallbacks)")
                print("CRITICAL ERROR: Failed to load model on any device or fallback!")

def estimate_tokens(prompt: str) -> Optional[int]:
    """
    Real tokenized length of the fully-formatted prompt, using the
    NPU pipeline's own tokenizer — not a chars/N heuristic. main.py's
    debug logging uses a chars/2.5 estimate for quick visibility, but
    that was already confirmed (see PROJECT_STATE.txt) to be an
    approximation that can meaningfully undercount real token usage.
    Since a real tokenizer is sitting right here, use it directly for
    the actual routing decision — the estimate that decides NPU vs.
    GPU should be as accurate as reasonably possible.

    Returns None if tokenization fails for any reason (e.g. an
    unexpected input_ids shape from a future openvino_genai version) —
    callers should treat None as "assume it fits, let NPU try normally"
    rather than raising, so a tokenizer API surprise degrades to
    today's existing behavior instead of breaking every request.
    """
    global pipe
    try:
        tokenizer = pipe.get_tokenizer()
        encoded = tokenizer.encode(prompt)
        input_ids = encoded.input_ids
        # input_ids is an ov::Tensor, NOT a subscriptable python
        # sequence -- encoded.input_ids[0] (previous code) raises
        # "'Tensor' object is not subscriptable". Tensor exposes shape
        # as [batch, seq_len]; the token count is the last dimension.
        # Tried in order of preference, each guarded independently so
        # a version difference in one path doesn't take down the rest:
        try:
            return int(input_ids.get_shape()[-1])
        except Exception:
            pass
        try:
            return int(input_ids.shape[-1])
        except Exception:
            pass
        # Last resort: pull the raw numpy array via .data and use its
        # shape/size directly.
        arr = input_ids.data
        return int(arr.shape[-1]) if hasattr(arr, "shape") else int(len(arr))
    except Exception as e:
        print(f"[token estimate] failed, assuming prompt fits NPU: {e}")
        return None


async def get_gpu_pipeline():
    """
    Returns the lazily-compiled GPU pipeline, compiling it on first
    call and reusing it after that. Thread/task-safe via
    gpu_compile_lock so two concurrent oversized requests can't race
    to compile it twice. Returns None if GPU compilation isn't
    possible on this machine (no compatible GPU, driver issue, etc.) —
    gpu_compile_failed is set so subsequent requests don't keep
    re-attempting a compile that's already known to fail, and instead
    fall straight back to trying NPU anyway (existing behavior, no
    worse than before this fallback existed).
    """
    global gpu_pipe, gpu_compile_failed, loaded_model_path

    if gpu_pipe is not None:
        return gpu_pipe
    if gpu_compile_failed:
        return None

    async with gpu_compile_lock:
        # Re-check inside the lock — another request may have already
        # compiled it (or failed it) while this one was waiting.
        if gpu_pipe is not None:
            return gpu_pipe
        if gpu_compile_failed:
            return None

        print(f"[GPU fallback] Compiling {loaded_model_path} onto 'GPU' (first oversized request)...")
        start_time = time.time()
        loop = asyncio.get_event_loop()
        try:
            # Compile in an executor thread — this can take well over
            # a minute (same order of magnitude as the NPU startup
            # compile) and must not block the event loop, which is
            # still serving other requests (short ones on NPU) while
            # this happens.
            gpu_pipe = await loop.run_in_executor(
                None, lambda: _build_pipeline(loaded_model_path, "GPU")
            )
            print(f"[GPU fallback] Compiled successfully in {time.time() - start_time:.2f}s. Staying warm for future overflow requests.")
            return gpu_pipe
        except Exception as e:
            print(f"[GPU fallback] Compilation failed, will not retry this process lifetime: {e}")
            gpu_compile_failed = True
            return None

def run_inference(target_pipe, prompt: str, config: ov_genai.GenerationConfig) -> str:
    """Wrapper function to run in executor. Takes the pipe explicitly
    now (NPU or lazily-compiled GPU) rather than always reading the
    global `pipe`, so the /api/chat routing logic can pick which
    device actually handles a given request."""
    return target_pipe.generate(prompt, config)


@app.get("/")
def read_root():
    global loaded_model_path
    return {
        "status": "NPU Offloader Server is running",
        "loaded_model": loaded_model_path if loaded_model_path else "None"
    }

@app.get("/api/health")
def health():
    # Truthful health: previously returned a static "ok" even when no
    # model had loaded (pipe is None) — the server looked healthy while
    # every /api/chat would 503. Report the actual pipeline state.
    # Stage 1 generation-speed window: throughput PROXY for cooldown
    # decisions. Field names deliberately avoid anything temperature-like —
    # this is chars/sec derived from real generation calls, not a sensor.
    snap = list(_gen_history)
    speeds = [r["chars_per_sec"] for r in snap]
    generation_speed_window = {
        "note": "throughput proxy (output chars/sec), NOT a temperature",
        "window_size": GEN_WINDOW_SIZE,
        "samples": len(snap),
        "avg_chars_per_sec": round(sum(speeds) / len(speeds), 2) if speeds else None,
        "min_chars_per_sec": round(min(speeds), 2) if speeds else None,
        "max_chars_per_sec": round(max(speeds), 2) if speeds else None,
        "oldest_sample_ts": snap[0]["ts"] if snap else None,
        "newest_sample_ts": snap[-1]["ts"] if snap else None,
    }

    return {
        "status": "ok" if pipe else "no_model",
        "model_loaded": pipe is not None,
        "loaded_model": loaded_model_path if loaded_model_path else None,
        # Actual prompt ceiling of the loaded pipeline (None = loaded
        # without a MAX_PROMPT_LEN constraint — unknown/default ceiling).
        # main.py derives its prompt target from this, so the ceiling
        # stays single-sourced in this file's MAX_PROMPT_LEN constant.
        "effective_max_prompt_len": effective_max_prompt_len,
        "safe_prompt_tokens": (
            max(0, effective_max_prompt_len - 644)
            if effective_max_prompt_len else None
        ),
        "generation_speed_window": generation_speed_window,
    }


class EstimateTokensRequest(BaseModel):
    messages: List[Dict[str, str]]


@app.post("/api/estimate_tokens")
def estimate_tokens_endpoint(payload: EstimateTokensRequest):
    """Real prompt-token count using the loaded model's own tokenizer.

    Tokenize-and-count ONLY: applies the same chat template /api/chat uses
    (add_generation_prompt=True, so generation-prompt tokens are included)
    and runs the tokenizer on CPU. No model inference, no NPU time — costs
    single-digit milliseconds for multi-thousand-token prompts. main.py
    uses this to right-size its prompt budget against MAX_PROMPT_LEN.
    """
    global pipe
    if not pipe:
        raise HTTPException(status_code=503, detail="Model pipeline is not initialized.")
    try:
        tokenizer = pipe.get_tokenizer()
        try:
            prompt = tokenizer.apply_chat_template(
                payload.messages, add_generation_prompt=True)
        except Exception as e:
            # Same manual-ChatML fallback /api/chat uses for Qwen.
            print(f"[estimate_tokens] template apply failed: {e}; formatting manually")
            prompt = ""
            for msg in payload.messages:
                prompt += f"<|im_start|>{msg.get('role', 'user')}\n{msg.get('content', '')}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        n = estimate_tokens(prompt)
        if n is None:
            raise HTTPException(status_code=500, detail="tokenization failed")
        return {"prompt_tokens": n}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"estimate_tokens error: {e}")

@app.post("/api/chat")
async def chat(payload: ChatRequest):
    global pipe
    if not pipe:
        raise HTTPException(status_code=503, detail="Model pipeline is not initialized or loaded.")
        
    # Format messages using OpenVINO GenAI tokenizer's chat template
    try:
        tokenizer = pipe.get_tokenizer()
        prompt = tokenizer.apply_chat_template(payload.messages, add_generation_prompt=True)
    except Exception as e:
        # Fallback to manual ChatML formatting (Qwen standard)
        print(f"Error applying template: {e}. Formatting manually.")
        prompt = ""
        for msg in payload.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    # Setup OpenVINO GenAI GenerationConfig
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 300  # Default matching main.py's num_predict limit
    
    if payload.options:
        opts = payload.options
        if "num_predict" in opts:
            config.max_new_tokens = int(opts["num_predict"])
            
        temp = opts.get("temperature", 1.0)
        top_p = opts.get("top_p", 1.0)
        top_k = opts.get("top_k", 0)
        
        # If sampling is configured, enable do_sample
        if temp != 1.0 or top_p != 1.0 or top_k > 0:
            config.do_sample = True
            config.temperature = float(temp)
            config.top_p = float(top_p)
            if top_k > 0:
                config.top_k = int(top_k)
                
        if "repeat_penalty" in opts:
            config.repetition_penalty = float(opts["repeat_penalty"])
        elif "repetition_penalty" in opts:
            config.repetition_penalty = float(opts["repetition_penalty"])
            
        if "presence_penalty" in opts:
            config.presence_penalty = float(opts["presence_penalty"])
        if "frequency_penalty" in opts:
            config.frequency_penalty = float(opts["frequency_penalty"])

    # --- Route to NPU or GPU based on real estimated prompt size ---
    # NPU has a hard compiled MAX_PROMPT_LEN=6144 — exceeding it fails
    # outright (500), no truncation attempted. Route oversized prompts
    # to the lazily-compiled GPU pipeline instead, which has no such
    # hard ceiling. See get_gpu_pipeline()/estimate_tokens() above for
    # the full reasoning.
    target_pipe = pipe
    target_device = "NPU"
    target_lock = npu_lock

    token_estimate = estimate_tokens(prompt)
    if token_estimate is not None and token_estimate > NPU_SAFE_PROMPT_TOKENS:
        if not GPU_FALLBACK_ENABLED:
            # Kill switch is on (known Vulkan driver issue on this
            # machine — see the GPU_FALLBACK_ENABLED comment above).
            # Skip the GPU compile attempt entirely and go straight to
            # trying NPU anyway, which fails cleanly with a catchable
            # 500 instead of risking a process-killing native crash.
            print(f"[routing] prompt estimated at {token_estimate} tokens, over NPU_SAFE_PROMPT_TOKENS={NPU_SAFE_PROMPT_TOKENS} — GPU fallback disabled (GPU_FALLBACK_ENABLED=False), attempting NPU anyway")
        else:
            print(f"[routing] prompt estimated at {token_estimate} tokens, over NPU_SAFE_PROMPT_TOKENS={NPU_SAFE_PROMPT_TOKENS} — attempting GPU fallback")
            candidate = await get_gpu_pipeline()
            if candidate is not None:
                target_pipe = candidate
                target_device = "GPU"
                target_lock = gpu_lock
            else:
                # GPU unavailable/failed to compile — fall back to trying
                # NPU anyway. Same outcome as before this fallback existed
                # (likely a 500 from NPU's own MAX_PROMPT_LEN check), not
                # worse.
                print("[routing] GPU fallback unavailable, attempting NPU anyway")

    # Run inference, serialized per-device (NPU and GPU requests use
    # separate locks so one device being busy doesn't block the other)
    t0 = time.time()
    async with target_lock:
        loop = asyncio.get_event_loop()
        try:
            # Generate in separate thread to prevent blocking FastAPI event loop
            reply = await loop.run_in_executor(None, run_inference, target_pipe, prompt, config)
        except Exception as e:
            print(f"Inference generation error on {target_device}: {e}")
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
            
    gen_time = time.time() - t0
    print(f"[{target_device} Server] Generated {len(reply)} chars in {gen_time:.2f}s")

    # Stage 1 observability: record throughput (recording only — no
    # enforcement). gen_time is always > 0 here in practice; the guard
    # covers the degenerate case instead of risking a divide-by-zero.
    _gen_history.append({
        "ts": time.time(),
        "gen_time": round(gen_time, 2),
        "output_chars": len(reply),
        "chars_per_sec": round(len(reply) / gen_time, 2) if gen_time > 0 else 0.0,
        "device": target_device,
    })
    
    # Return response mirroring Ollama's format
    return {
        "message": {
            "role": "assistant",
            "content": reply
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=11435)
