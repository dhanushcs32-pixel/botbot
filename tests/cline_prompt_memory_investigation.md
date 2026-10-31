# Prompt for Cline

I need you to investigate a real problem, then propose (don't yet build,
unless I say so) the lowest-hardware-strain fix — and you're allowed to
conclude the Obsidian/LLM-wiki idea below is NOT the right answer. Decide
based on evidence, not on the idea I'm most recently excited about.

## Hardware context — the binding constraint on everything

This stack runs a single Qwen2.5-7B-Instruct int4 model on one Intel NPU,
serialized behind `NPUThrottle` (main.py) — no concurrency, one generation
at a time, 45–130s observed per NPU call. RAM is already tight (that's why
hybrid search was built to run in ~80ms with no extra NPU calls instead of
an LLM-based reranker). Every proposal below must be evaluated primarily
on: how much NPU time does this cost, and when does it run (live chat path
vs. offline/background vs. manual-trigger). Prefer zero-NPU-cost solutions
over cheap-NPU solutions over expensive-NPU solutions, in that order,
UNLESS quality would clearly suffer too much — flag that tradeoff
explicitly rather than silently picking the cheap option.

## The actual problem (please verify, don't assume)

Running `analyze_sufficiency_log.py` (385 logged turns) surfaced 11
candidate confabulations. On manual review, most are low-stakes, but two
patterns are real:

1. One reply gave confident, specific, WRONG advice (treated a
   crypto/debt-timing message as a stalking-victim scenario: "block their
   number," "freeze bank accounts," "report to authorities" — none of
   which the user's message was about).
2. Multiple replies to specific, emotionally loaded, repeated content
   (grief over named people mentioned across several sessions) came back
   generic/deflecting rather than grounded in what was actually said.

Hypothesis A (mine, unverified): this is a prompt-token-budget problem —
`npu_server.py`'s `MAX_PROMPT_LEN=6144` may be silently truncating
retrieved context before it reaches the model.

Hypothesis B (mine, unverified): this is NOT primarily a retrieval
problem — it's a generation/style behavior problem. The model has
"enough" context in some of these cases but still produces generic output.

Hypothesis C (mine, unverified): summaries in `memory.db` are themselves
too lossy/generic by the time they're written — i.e. the compression step
(summarizer.py's rolling/session summarization) is where specificity gets
lost, independent of retrieval or prompt budget.

## What I want you to actually do

### Step 1 — Instrument before theorizing
Add lightweight, low-cost instrumentation (no extra NPU calls) that logs,
per turn: total assembled prompt tokens, tokens per section (system
prompt / RAG personal / RAG reference / recent turns / summaries), and
whether/what got trimmed by the hard char budget in mode_router.py. This
tests Hypothesis A directly and cheaply — it's just token counting, not
inference.

### Step 2 — Pull the actual prompts for the two real confabulation cases
For the crypto/debt message and at least 2 of the grief-related messages
from the sufficiency log, reconstruct exactly what was assembled and sent
to the NPU for that turn (you may need a debug flag or a one-off script —
don't touch production code paths to do this). Determine: was the
relevant context (e.g., prior mentions of the named people, or the actual
topic of the crypto message) present in the assembled prompt at all? If
yes, Hypothesis A is wrong and B/C become more likely. If no, check
whether it was retrieved but trimmed, or never retrieved at all — those
have different fixes (budget vs. retrieval).

### Step 3 — Check summary quality (Hypothesis C)
Pull the actual `session_summaries`/`rolling_summaries` rows covering the
sessions where the named people appear repeatedly. Are the summaries
themselves generic ("user discussed missing friends") or specific ("user
named [X, Y] repeatedly across N sessions, expressed [specific stated
feelings], mentioned [specific stated facts]")? This is free to check —
no NPU calls, just reading memory.db.

### Step 4 — Decide, with evidence, what's actually worth building

You have several options on the table. Pick the one the evidence from
Steps 1–3 actually supports, or propose something else if none fit. Rank
by hardware strain, cheapest first, and only reach for something more
expensive if the evidence says a cheaper fix won't work:

  a) **Prompt budget fix** (raise MAX_PROMPT_LEN, or smarter trimming that
     protects person/entity-tagged content from truncation first) — cheap,
     config-level, zero new NPU calls, only fixes the problem if Step 2
     shows truncation is actually happening.

  b) **Summarization prompt fix** (summarizer.py's instructions demand
     more specificity/named entities preserved) — cheap, no new
     infrastructure, only fixes it if Step 3 shows summaries are the lossy
     point. Costs nothing extra at runtime; the summarization NPU calls
     already happen today either way.

  c) **Retrieval fix** (e.g., ensure person-tagged content is
     boosted/guaranteed-included for messages mentioning known people,
     using the known-people extraction that already exists in rag.py) —
     cheap, no new NPU calls, works alongside hybrid search already
     shipped.

  d) **LLM-wiki / Obsidian-style compiled memory** (the idea discussed
     earlier: raw/ → LLM-compiled per-entity .md wiki pages, viewable in
     Obsidian, lint pass for orphan/missing entities) — real NPU cost for
     ingest+lint, MUST be offline/manual-trigger only, never in the live
     chat path. Only propose this if a/b/c are evidently insufficient —
     e.g., if the problem is that no amount of better-retrieval-of-raw-
     chunks would help because durable cross-session understanding
     genuinely needs to be pre-compiled, not re-derived per turn. If you
     recommend this, also specify: trigger cadence (manual only, at least
     initially), scope per run (bounded, not "recompile everything"), and
     lint design (single pass with full wiki context, NOT pairwise
     comparisons — avoid combinatorial NPU cost growth as the wiki grows).

  e) Anything else the evidence points to that isn't listed above.

### Step 5 — Report back, do not implement yet
Give me: what Steps 1–3 actually found (with the real numbers/examples,
not summaries of summaries), which hypothesis(es) the evidence supports,
your ranked recommendation with estimated NPU cost/frequency for each
viable option, and an explicit statement of any quality-vs-hardware-strain
tradeoff if the cheapest fix isn't the highest-quality one. I will decide
what to build after reading this.

Do not modify semantic_style_check.py, followup.py, or anything already
verified in PROJECT_STATE.txt without flagging it first. Do not start
Phase 2 (agentic RAG) — that was already assessed and deferred.
