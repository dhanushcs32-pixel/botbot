"""Hybrid search regression test (dense + BM25 via RRF).

Run:  cd C:\\botbot && .\\venv\\Scripts\\python.exe tests\\test_hybrid_search.py

Part A: RRF fusion math (pure logic, no ChromaDB/embeddings).
Part B: flag-off fallback returns dense results untouched.
Part C: integration against the REAL personal_data collection (read-only):
  picks a rare corpus token, queries it, and asserts the exact-token chunk
  surfaces via hybrid fusion even when pure dense ranks it poorly.

Never prints chunk document content — only ids, ranks, and scores
(the collection holds private conversation data).
"""
import io
import os
import sys
import time
from pathlib import Path

# The embedding model is cached locally (production queries use it); force
# offline mode so sentence-transformers doesn't ping huggingface.co at load
# time — an offline/failed DNS box otherwise poisons the client and the
# whole collection becomes unreachable for the test.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    ok = ok and cond


# ---------- Part A: RRF math ----------
fused = rag.rrf_fuse([["a", "b", "c"], ["c", "d"]])
# c: 1/61+1/61 (top of both), a: 1/61, b: 1/62, d: 1/62 (b before d: insertion order on tie)
check("RRF: dual-top item ranks first", fused[0] == "c", f"{fused}")
check("RRF: both lists' items present, ordered by fused score",
      fused == ["c", "a", "b", "d"], f"{fused}")
check("RRF: empty input -> empty output", rag.rrf_fuse([[], []]) == [])

# ---------- Part B: flag-off fallback ----------
real_flag = rag.HYBRID_SEARCH_ENABLED
rag.HYBRID_SEARCH_ENABLED = False
docs, metas = rag._hybrid_fuse_chunks(
    "personal", "anything", 5,
    ["id1", "id2"], ["doc1", "doc2"], [{"m": 1}, {"m": 2}],
    predicates=[rag._bm25_predicate_none])
rag.HYBRID_SEARCH_ENABLED = real_flag
check("flag off: dense results returned untouched (no index access)",
      docs == ["doc1", "doc2"] and metas == [{"m": 1}, {"m": 2}])

# ---------- Part C: integration on the real collection ----------
try:
    collection = rag.get_collection()
    count = collection.count()
except Exception as e:
    print(f"[SKIP] personal collection unavailable: {e}")
    sys.exit(0 if ok else 1)

if count == 0:
    print("[SKIP] personal collection is empty")
    sys.exit(0 if ok else 1)

try:
    import psutil
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / (1024 * 1024)
except ImportError:
    proc = None
    rss_before = None

t0 = time.time()
entry = rag._bm25_get_index("personal")
build_secs = time.time() - t0
if proc is not None:
    rss_after = proc.memory_info().rss / (1024 * 1024)
    mem_note = f"RSS +{rss_after - rss_before:.0f} MB (now {rss_after:.0f} MB)"
else:
    mem_note = "psutil not installed — RSS not measured"
check("BM25 index built over the real collection",
      entry is not None and entry["bm25"] is not None,
      f"{count} chunks, {build_secs:.1f}s, {mem_note}")

# Pick a rare target token + its chunk: sample pages, take a long token,
# confirm it's rare via the BM25 document frequencies.
import random
random.seed(42)
target_id = None
target_token = None
for attempt in range(6):
    offset = random.randrange(0, max(count - 200, 1))
    page = collection.get(include=["documents"], limit=200, offset=offset)
    for cid, doc in zip(page.get("ids", []), page.get("documents", [])):
        if not doc:
            continue
        toks = [t for t in rag._bm25_tokenize(doc) if len(t) >= 8]
        random.shuffle(toks)
        for tok in toks[:4]:
            df = sum(1 for tf in entry["bm25"].doc_freqs if tok in tf)
            if df == 1:
                target_id, target_token = cid, tok
                break
        if target_id:
            break
    if target_id:
        break

if not target_id:
    print("[SKIP] no rare (df==1) token found in sampled pages; "
          "integration assertion needs a distinctive token")
    sys.exit(0 if ok else 1)

query = f"what was {target_token} about exactly"
print(f"probe: rare token {target_token!r} (df=1), target chunk {target_id}")

# Dense-only baseline (no metadata filters, mirroring the final fallback stage):
t0 = time.time()
dense = collection.query(query_texts=[query], n_results=10,
                         include=["documents"])
dense_secs = time.time() - t0
dense_ids = dense.get("ids", [[]])[0]
dense_rank = dense_ids.index(target_id) + 1 if target_id in dense_ids else None

# Hybrid: BM25 rank + RRF against the dense list.
t0 = time.time()
bm25_ranked = rag._bm25_rank("personal", query, rag._bm25_predicate_never_bot, 30)
bm25_secs = time.time() - t0
bm25_rank = bm25_ranked.index(target_id) + 1 if target_id in bm25_ranked else None
fused = rag.rrf_fuse([dense_ids, bm25_ranked])[:10]
fused_rank = fused.index(target_id) + 1 if target_id in fused else None

print(f"dense: rank={dense_rank} ({dense_secs * 1000:.0f}ms) | "
      f"bm25: rank={bm25_rank} ({bm25_secs * 1000:.0f}ms) | "
      f"fused: rank={fused_rank}")

check("BM25 finds the exact-token chunk in top-5",
      bm25_rank is not None and bm25_rank <= 5,
      f"rank={bm25_rank}")
check("fused ranking beats or matches dense for the exact-token chunk",
      fused_rank is not None and (dense_rank is None or fused_rank <= dense_rank)
      and fused_rank <= 5,
      f"fused={fused_rank} vs dense={dense_rank}")

# End-to-end wiring: the real search function runs clean with hybrid on.
out = rag.search_personal_data(query, top_k=5)
check("search_personal_data end-to-end returns a non-empty context string",
      isinstance(out, str) and len(out) > 0, f"{len(out)} chars")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
