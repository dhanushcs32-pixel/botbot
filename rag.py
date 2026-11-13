"""
Bot Bot — RAG Module (Part 6 & 7)

Sets up a local ChromaDB vector database, indexes personal data
(conversations, journals, notes) from the personal_data folder,
and provides a search function to retrieve relevant context for
any given user message.

Everything runs 100% locally — embeddings are generated on-device
using sentence-transformers, no cloud calls involved.
"""

import os
import re
import time
from pathlib import Path
from datetime import date, datetime, timedelta
from dateutil import parser as dateutil_parser
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from laya_engine import rerank_and_filter_chunks, LAYA_ENABLED, is_laya_available


PERSONAL_DATA_DIR = Path(__file__).parent / "personal_data"
CHROMA_STORE_DIR = Path(__file__).parent / "chroma_store"

# Finished bot sessions get exported here (summary header + full transcript,
# one .txt per session) by conversation_export.py, and indexed into the same
# `collection` as everything else, tagged source_type="bot_conversation" so
# they can be searched separately from other-people's-chat data.
BOT_CONVERSATIONS_DIR = PERSONAL_DATA_DIR / "bot_conversations"

MESSAGES_PER_CHUNK = 10   # how many consecutive messages form one chunk
MESSAGE_OVERLAP = 3       # messages repeated between consecutive chunks

# Size caps for the personal-data chunking fix
MAX_PERSONAL_MESSAGE_CHARS = 1200   # cap on a single paragraph-fallback "message"
PERSONAL_MESSAGE_OVERLAP_CHARS = 150
MAX_PERSONAL_CHUNK_CHARS = 3000     # cap on a whole chunk (up to MESSAGES_PER_CHUNK messages)
MAX_PERSONAL_RESULT_CHARS = 8000    # DEFAULT fallback cap across all top_k results combined
                                     # (top_k=8 default). Fix 4: search_personal_data() now
                                     # accepts a max_result_chars param so callers (main.py,
                                     # via RETRIEVAL_POLICY's per-tier personal_max_chars in
                                     # mode_router.py) can pass a smaller cap for cheaper modes
                                     # (e.g. conversational) instead of always paying for 8000.
MAX_EXPANDED_SESSION_CHARS = 6000   # cap per expanded full bot_conversation session file

# Matches WhatsApp export lines like:
# "12/06/2026, 14:32 - Name: message text"
WHATSAPP_LINE_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?:\s*[APap][Mm])?)\s*-\s"
)

# ----------------------------------------------------------------------
# LAZY LOADING SYSTEM
# Prevents Uvicorn from crashing during boot by delaying the model 
# download/loading until the moment a search is actually performed.
# ----------------------------------------------------------------------
_embedding_fn = None
_collection = None
_reference_collection = None
_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_STORE_DIR))
    return _chroma_client


def get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        print("[RAG] Loading embedding model...")
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embedding_fn


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_chroma_client().get_or_create_collection(
            name="personal_data",
            embedding_function=get_embedding_fn()
        )
    return _collection


def get_reference_collection():
    global _reference_collection
    if _reference_collection is None:
        _reference_collection = get_chroma_client().get_or_create_collection(
            name="reference_data",
            embedding_function=get_embedding_fn()
        )
    return _reference_collection
# ----------------------------------------------------------------------


def split_into_messages(text: str) -> list[dict]:
    """
    Splits raw chat export text into individual messages. Handles
    WhatsApp's format where a new message starts with a date/time
    line, and multi-line messages continue until the next one.
    Falls back to paragraph-based splitting for non-WhatsApp text.
    """
    lines = text.splitlines()
    messages = []
    current_lines = []
    current_date = None
    current_time = None

    is_whatsapp_style = any(WHATSAPP_LINE_PATTERN.match(line) for line in lines[:20])

    if is_whatsapp_style:
        for line in lines:
            match = WHATSAPP_LINE_PATTERN.match(line)
            if match:
                if current_lines:
                    messages.append({
                        "date": current_date,
                        "time": current_time,
                        "text": "\n".join(current_lines).strip()
                    })
                current_date, current_time = match.group(1), match.group(2)
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            messages.append({
                "date": current_date,
                "time": current_time,
                "text": "\n".join(current_lines).strip()
            })
        return [m for m in messages if m["text"]]

    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [text.strip()]
    paragraphs = []
    for p in raw_paragraphs:
        paragraphs.extend(_split_oversized_paragraph(
            p, max_chars=MAX_PERSONAL_MESSAGE_CHARS, overlap=PERSONAL_MESSAGE_OVERLAP_CHARS
        ))
    return [{"date": None, "time": None, "text": p} for p in paragraphs]


def _parse_whatsapp_date(date_str: str) -> date | None:
    if not date_str:
        return None
    try:
        return dateutil_parser.parse(date_str, dayfirst=True).date()
    except (ValueError, OverflowError):
        return None


def _epoch_for(d: date) -> int:
    return int(datetime(d.year, d.month, d.day).timestamp())


def _date_range_for(messages: list[dict]) -> dict:
    raw_dates = [m["date"] for m in messages if m.get("date")]
    if not raw_dates:
        return {"date_range": "", "date_start_epoch": None, "date_end_epoch": None}

    first_str, last_str = raw_dates[0], raw_dates[-1]
    date_range = first_str if first_str == last_str else f"{first_str} - {last_str}"

    parsed = [d for d in (_parse_whatsapp_date(s) for s in raw_dates) if d is not None]
    if not parsed:
        return {"date_range": date_range, "date_start_epoch": None, "date_end_epoch": None}

    return {
        "date_range": date_range,
        "date_start_epoch": _epoch_for(min(parsed)),
        "date_end_epoch": _epoch_for(max(parsed))
    }


def chunk_messages(messages: list[dict], size: int = MESSAGES_PER_CHUNK,
                    overlap: int = MESSAGE_OVERLAP,
                    max_chars: int = MAX_PERSONAL_CHUNK_CHARS) -> list[dict]:
    if not messages:
        return []

    def make_chunk(slice_: list[dict]) -> dict:
        date_info = _date_range_for(slice_)
        return {
            "text": "\n".join(m["text"] for m in slice_),
            **date_info
        }

    chunks = []
    start = 0
    n = len(messages)
    while start < n:
        end = start
        char_count = 0
        while end < n and (end - start) < size:
            next_len = len(messages[end]["text"]) + 1  # +1 for the join newline
            if char_count + next_len > max_chars and end > start:
                break
            char_count += next_len
            end += 1
        if end == start:
            end = start + 1

        chunks.append(make_chunk(messages[start:end]))
        if end >= n:
            break

        step = max(end - start - overlap, 1)
        start += step

    return chunks


PLATFORM_MARKER_PATTERN = re.compile(r"(whatsapp|insta)", re.IGNORECASE)
NOTE_FILE_PREFIXES = ("journal", "notes", "note", "diary", "todo")


def extract_person_key(filename: str) -> str:
    """
    Derives a person key from a personal_data filename, e.g.
    "Name1Whatsapp.txt" / "NameInsta.txt" -> "name".
    Notes, journals, diaries, and todos return "" (they are personal notes,
    not person chats).
    """
    name = re.sub(r"\.txt$", "", filename, flags=re.IGNORECASE)
    stem_lower = name.lower()
    for prefix in NOTE_FILE_PREFIXES:
        if stem_lower.startswith(prefix):
            return ""

    name = re.sub(r"^(?:chat_with_|message(?:_from)?_)", "", name, flags=re.IGNORECASE)
    name = PLATFORM_MARKER_PATTERN.sub("", name)
    name = re.sub(r"^[\d_]+", "", name)
    name = re.sub(r"[\d_]+$", "", name)
    return name.lower().strip("_")


def index_single_file(file_path: Path) -> int:
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin1")

    messages = split_into_messages(text)
    chunk_dicts = chunk_messages(messages)
    if not chunk_dicts:
        return 0

    relative_source = str(file_path.relative_to(PERSONAL_DATA_DIR))
    safe_id_base = re.sub(r"[^a-zA-Z0-9]", "_", relative_source)

    is_bot_conversation = BOT_CONVERSATIONS_DIR in file_path.parents
    if is_bot_conversation:
        source_type = "bot_conversation"
        person_key = ""
    else:
        person_key = extract_person_key(file_path.name)
        source_type = "personal_chat" if person_key else "notes"

    chunks = [c["text"] for c in chunk_dicts]
    ids = [f"{safe_id_base}_{i}" for i in range(len(chunks))]

    def build_metadata(c: dict) -> dict:
        meta = {
            "source": relative_source,
            "person": person_key,
            "date_range": c["date_range"],
            "source_type": source_type
        }
        if c["date_start_epoch"] is not None:
            meta["date_start_epoch"] = c["date_start_epoch"]
            meta["date_end_epoch"] = c["date_end_epoch"]
        return meta

    metadatas = [build_metadata(c) for c in chunk_dicts]

    MAX_BATCH = 4000
    try:
        get_collection().delete(where={"source": relative_source})
    except Exception:
        pass

    for batch_start in range(0, len(chunks), MAX_BATCH):
        batch_end = batch_start + MAX_BATCH
        get_collection().upsert(
            ids=ids[batch_start:batch_end],
            documents=chunks[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end]
        )

    print(f"Indexed: {relative_source} ({len(chunks)} chunks, source_type={source_type})")
    invalidate_known_people_cache()
    return len(chunks)


def index_personal_data():
    if not PERSONAL_DATA_DIR.exists():
        print(f"ERROR: {PERSONAL_DATA_DIR} does not exist. "
              f"Create the folder and add .txt files first.")
        return

    txt_files = list(PERSONAL_DATA_DIR.rglob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {PERSONAL_DATA_DIR}. Nothing to index.")
        return

    total_chunks = 0
    for file_path in txt_files:
        if file_path == MY_STYLE_PATH:
            continue  
        total_chunks += index_single_file(file_path)

    invalidate_known_people_cache()
    print(f"\nDone. Indexed {len(txt_files)} file(s), {total_chunks} total chunks.")


MY_STYLE_PATH = PERSONAL_DATA_DIR / "my_style_template.txt"
_style_cache = {"mtime": None, "content": ""}


def get_style_profile() -> str:
    """Reads the user's personality template to constrain the bot's behavior."""
    base_dir = Path(__file__).parent
    template_path = base_dir / "personal_data" / "my_style_template.txt"
    
    if not template_path.exists():
        print(f"[warning] Style profile not found at {template_path}")
        return ""
    
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return f"### BEHAVIORAL RULES ###\n{content}\n"
            return ""
    except Exception as e:
        print(f"[warning] Failed to read my_style_template.txt: {e}")
        return ""


def keyword_search(query: str, context_lines: int = 8, max_results: int = 15) -> str:
    query = query.strip()
    if not query or len(query) < 2:
        return ""

    query_lower = query.lower()
    txt_files = list(PERSONAL_DATA_DIR.rglob("*.txt"))
    matches = []

    for file_path in txt_files:
        if file_path == MY_STYLE_PATH:
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="latin1")

        lines = text.splitlines()
        relative_source = str(file_path.relative_to(PERSONAL_DATA_DIR))

        for i, line in enumerate(lines):
            if query_lower in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                snippet = "\n".join(lines[start:end])
                matches.append((relative_source, snippet))

                if len(matches) >= max_results:
                    break

        if len(matches) >= max_results:
            break

    if not matches:
        return ""

    tagged = [f"[From: {source}]\n{snippet}" for source, snippet in matches]
    return "\n---\n".join(tagged)


REFERENCE_DATA_DIR = PERSONAL_DATA_DIR / "reference"

MAX_REFERENCE_CHUNK_CHARS = 1200
REFERENCE_CHUNK_OVERLAP_CHARS = 150
MAX_REFERENCE_RESULT_CHARS = 4000  


def _split_oversized_paragraph(paragraph: str, max_chars: int = MAX_REFERENCE_CHUNK_CHARS,
                                overlap: int = REFERENCE_CHUNK_OVERLAP_CHARS) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars - overlap):
                pieces.append(sentence[i:i + max_chars])
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars:
            if current:
                pieces.append(current)
            current = (current[-overlap:] + " " + sentence).strip() if current else sentence
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


# Matches the book's confirmed heading format, e.g. "CHAPTER 3:   TITLE HERE"
# (all caps, colon, extra spaces). Confirmed each heading lands as the start
# of its own \n\n-delimited paragraph.
CHAPTER_HEADING_PATTERN = re.compile(r"^CHAPTER\s+(\d{1,2}):\s*(.+)$")

# Matches a line containing ONLY a worksheet numbered-blank marker, e.g. a
# bare "3." on its own line. Deliberately line-level (not paragraph-level):
# confirmed these often share a \n\n-delimited paragraph block with real
# prose (PDF-to-text conversion didn't reliably insert blank lines between
# them), so dropping whole paragraphs would risk losing real content.
WORKSHEET_BLANK_LINE_PATTERN = re.compile(r"^\s*\d{1,2}\.\s*$")


def _strip_worksheet_lines(text: str) -> str:
    """
    Removes blank numbered-list lines (worksheet/fill-in-the-blank noise)
    line-by-line, before any paragraph splitting happens, so real prose
    sharing a paragraph with worksheet lines is preserved.
    """
    lines = text.splitlines()
    kept = [line for line in lines if not WORKSHEET_BLANK_LINE_PATTERN.match(line)]
    return "\n".join(kept)


def _split_into_chapters(text: str) -> list[dict]:
    """
    Splits reference text into chapters using the confirmed
    "CHAPTER <N>:   <Title>" heading format, each expected at the start of
    its own \n\n-delimited paragraph. Any text before the first heading (or
    all text, if no headings are found) is kept as an untitled chapter
    (chapter_number=None) so non-chapterized reference files still index
    instead of being dropped.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []

    chapters = []
    current = {"chapter_number": None, "chapter_title": None, "paragraphs": []}

    for p in raw_paragraphs:
        # Headings share their \n\n-delimited paragraph block with
        # following body text (same PDF-to-text quirk as the worksheet
        # lines), so only test the paragraph's FIRST line against the
        # pattern rather than the whole (possibly multi-line) paragraph.
        first_line, _, rest = p.partition("\n")
        match = CHAPTER_HEADING_PATTERN.match(first_line.strip())
        if match:
            if current["paragraphs"]:
                chapters.append(current)
            current = {
                "chapter_number": int(match.group(1)),
                "chapter_title": match.group(2).strip().title(),
                "paragraphs": []
            }
            # Preserve any body text that shared the paragraph with the
            # heading instead of discarding it.
            if rest.strip():
                current["paragraphs"].append(rest.strip())
        else:
            current["paragraphs"].append(p)

    if current["paragraphs"]:
        chapters.append(current)

    return chapters


def _chunk_reference_text(text: str) -> list[dict]:
    """
    Chapter-aware chunking: strips worksheet noise first, splits the
    remaining text on the confirmed "CHAPTER <N>:" headings, then runs the
    existing paragraph/sentence chunking WITHIN each chapter. Returns chunk
    dicts (not plain strings) carrying chapter_number/chapter_title so
    index_reference_data() can attach them as metadata.
    """
    text = _strip_worksheet_lines(text)
    chapters = _split_into_chapters(text)

    chunks = []
    for chapter in chapters:
        for p in chapter["paragraphs"]:
            for piece in _split_oversized_paragraph(p):
                chunks.append({
                    "text": piece,
                    "chapter_number": chapter["chapter_number"],
                    "chapter_title": chapter["chapter_title"]
                })
    return chunks


def index_reference_data():
    if not REFERENCE_DATA_DIR.exists():
        print(f"No reference folder found at {REFERENCE_DATA_DIR}. Skipping.")
        return

    txt_files = list(REFERENCE_DATA_DIR.rglob("*.txt"))
    if not txt_files:
        print("No reference .txt files found. Skipping.")
        return

    total_chunks = 0
    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="latin1")

        chunk_dicts = _chunk_reference_text(text)
        if not chunk_dicts:
            continue

        relative_source = str(file_path.relative_to(REFERENCE_DATA_DIR))
        safe_id_base = re.sub(r"[^a-zA-Z0-9]", "_", relative_source)

        try:
            get_reference_collection().delete(where={"source": relative_source})
        except Exception:
            pass  

        ids = [f"ref_{safe_id_base}_{i}" for i in range(len(chunk_dicts))]
        documents = [c["text"] for c in chunk_dicts]

        def build_reference_metadata(c: dict) -> dict:
            # Chroma rejects None metadata values, so chapter fields are
            # only included when a heading was actually matched.
            meta = {"source": relative_source}
            if c["chapter_number"] is not None:
                meta["chapter_number"] = c["chapter_number"]
            if c["chapter_title"]:
                meta["chapter_title"] = c["chapter_title"]
            return meta

        metadatas = [build_reference_metadata(c) for c in chunk_dicts]

        get_reference_collection().upsert(ids=ids, documents=documents, metadatas=metadatas)
        total_chunks += len(chunk_dicts)
        n_with_chapter = sum(1 for c in chunk_dicts if c["chapter_number"] is not None)
        print(f"Indexed reference: {relative_source} "
              f"({len(chunk_dicts)} chunks, {n_with_chapter} chapter-tagged)")

    print(f"\nReference data done. {len(txt_files)} file(s), {total_chunks} chunks.")


def build_reference_excerpt_label(meta: dict) -> str:
    """Mirrors build_excerpt_label()'s pattern for personal_data results,
    so retrieved therapy content is traceable to its source chapter."""
    meta = meta or {}
    chapter_number = meta.get("chapter_number")
    chapter_title = meta.get("chapter_title")
    source = meta.get("source", "unknown")

    if chapter_number is not None and chapter_title:
        return f"Chapter {chapter_number}: {chapter_title}"
    if chapter_number is not None:
        return f"Chapter {chapter_number}"
    return f"Reference ({source})"


# ---------------------------------------------------------------------
# Hybrid retrieval: dense (ChromaDB/MiniLM) + sparse (BM25) fused via
# Reciprocal Rank Fusion (RRF).
#
# Why: dense embedding search handles meaning ("something about exams
# during that trip") but is weak on exact rare tokens — proper names,
# dates-as-text, unique places. BM25 is the complement. RRF (score =
# sum of 1/(k + rank), k=60, the standard damping constant) merges the
# two rank lists without comparing incompatible score scales.
#
# Config: set HYBRID_SEARCH_ENABLED = False to fall back to dense-only
# (previous behavior, byte-for-byte path).
#
# Memory/build notes: the BM25 index is built lazily on first hybrid
# query by paginating the collection (BM25_PAGE_SIZE — same limit as
# get_known_people, ChromaDB's SQLite backend errors on large unpaginated
# gets) and caches ONLY ids + metadata + BM25's internal term structures;
# chunk documents are NOT retained — they are fetched from ChromaDB by id
# on demand. It rebuilds automatically when collection.count() changes
# (new exports / prunes). Same-count staleness is an ACCEPTED, documented
# assumption, not an oversight: this collection is append-mostly — journals/
# chat exports are added via manual re-index runs, the only deletion path is
# the rare manual tests/prune_orphaned_index.py maintenance script, and any
# manual re-index of *edited* files (upsert, same count) is covered no later
# than the next count change or process restart (the cache is in-memory and
# this stack restarts via start_botbot.bat). If same-count edits ever become
# routine, replace the count check with content-hash or TTL invalidation —
# flag it before changing.
# ---------------------------------------------------------------------
HYBRID_SEARCH_ENABLED = True
RRF_K = 60
BM25_PAGE_SIZE = 1000
BM25_FETCH_MULTIPLIER = 3  # sparse candidates fetched per final top_k

_BM25_TOKEN_RE = re.compile(r"[a-z0-9']+")
_BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "if", "in", "is", "it",
    "its", "me", "my", "no", "not", "of", "on", "or", "our", "she", "so",
    "that", "the", "their", "them", "then", "there", "they", "this", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "you", "your",
}

_bm25_cache: dict = {}  # kind -> {"count", "bm25", "ids", "metas"}


def _bm25_tokenize(text: str) -> list[str]:
    return [
        t for t in _BM25_TOKEN_RE.findall(text.lower())
        if t not in _BM25_STOPWORDS and len(t) > 1
    ]


def _bm25_get_index(kind: str):
    """Lazy-build (or refresh) the BM25 index for a collection."""
    collection = get_collection() if kind == "personal" else get_reference_collection()
    count = collection.count()
    cached = _bm25_cache.get(kind)
    if cached and cached["count"] == count:
        return cached

    import time as _time
    t0 = _time.time()
    ids, metas, corpus = [], [], []
    offset = 0
    while offset < count:
        page = collection.get(
            include=["documents", "metadatas"],
            limit=BM25_PAGE_SIZE, offset=offset,
        )
        page_ids = page.get("ids", [])
        page_docs = page.get("documents", [])
        page_metas = page.get("metadatas", [])
        if not page_ids:
            break
        ids.extend(page_ids)
        metas.extend(page_metas)
        corpus.extend(page_docs)
        offset += BM25_PAGE_SIZE

    tokenized = [_bm25_tokenize(doc) for doc in corpus]
    del corpus
    bm25 = BM25Okapi(tokenized) if tokenized else None
    entry = {"count": count, "bm25": bm25, "ids": ids, "metas": metas}
    _bm25_cache[kind] = entry
    print(f"[rag] BM25 index built for '{kind}': {count} chunks in "
          f"{_time.time() - t0:.1f}s")
    return entry


def _bm25_rank(kind: str, query: str, predicate, fetch_n: int) -> list[str]:
    """Top fetch_n chunk ids matching predicate, ranked by BM25 score."""
    entry = _bm25_get_index(kind)
    if entry["bm25"] is None:
        return []
    scores = entry["bm25"].get_scores(_bm25_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked = []
    for i in order:
        if scores[i] <= 0:
            break  # scores are sorted; everything after is a zero match
        if predicate(entry["metas"][i] or {}):
            ranked.append(entry["ids"][i])
            if len(ranked) >= fetch_n:
                break
    return ranked


def rrf_fuse(rank_lists: list, k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion over one or more ranked id lists."""
    scores: dict = {}
    for ranks in rank_lists:
        for pos, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + pos + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _bm25_predicate_bot_only(meta: dict) -> bool:
    return (meta or {}).get("source_type") == "bot_conversation"


def _bm25_predicate_never_bot(meta: dict) -> bool:
    return (meta or {}).get("source_type") != "bot_conversation"


def _bm25_predicate_none(meta: dict) -> bool:
    return True


def _bm25_predicate_personal(mentioned_person, date_range) -> list:
    """Cascade of metadata predicates mirroring search_personal_data's
    dense fallback stages: person+date, then person, then base."""
    def strict(meta):
        if not _bm25_predicate_never_bot(meta):
            return False
        if mentioned_person and meta.get("person") != mentioned_person:
            return False
        if date_range:
            start_epoch = _epoch_for(date_range[0])
            end_epoch = _epoch_for(date_range[1]) + 86400
            ds = meta.get("date_start_epoch")
            de = meta.get("date_end_epoch")
            if ds is None or de is None:
                return False  # mirrors Chroma where: missing field = no match
            if not (ds <= end_epoch and de >= start_epoch):
                return False
        return True

    def relaxed(meta):
        if not _bm25_predicate_never_bot(meta):
            return False
        if mentioned_person and meta.get("person") != mentioned_person:
            return False
        return True

    return [strict, relaxed, _bm25_predicate_never_bot]


def _hybrid_fuse_chunks(kind: str, query: str, top_k: int,
                        dense_ids: list, dense_docs: list, dense_metas: list,
                        predicates) -> tuple:
    """Fuse dense results with BM25 via RRF; returns (docs, metas) in fused
    order, truncated to top_k. Dense results are rank 1..n as passed in;
    sparse candidates come from _bm25_rank over the metadata predicates
    (first predicate that yields enough candidates wins — mirrors the
    dense fallback cascade). Fails open to dense-only on any error."""
    if not HYBRID_SEARCH_ENABLED:
        return list(dense_docs), list(dense_metas)

    fetch_n = max(top_k * BM25_FETCH_MULTIPLIER, len(dense_ids))
    if predicates is None:
        predicates = [_bm25_predicate_none]

    bm25_ranked: list = []
    for pred in predicates:
        bm25_ranked = _bm25_rank(kind, query, pred, fetch_n)
        if len(bm25_ranked) >= fetch_n:
            break

    fused_ids = rrf_fuse([list(dense_ids), bm25_ranked])[:top_k]
    if not fused_ids:
        return list(dense_docs), list(dense_metas)

    dense_map = dict(zip(dense_ids, zip(dense_docs, dense_metas)))
    missing = [cid for cid in fused_ids if cid not in dense_map]
    fetched = {}
    if missing:
        collection = get_collection() if kind == "personal" else get_reference_collection()
        res = collection.get(ids=missing, include=["documents", "metadatas"])
        fetched = {
            cid: (doc, meta)
            for cid, doc, meta in zip(
                res.get("ids", []), res.get("documents", []), res.get("metadatas", []))
        }

    docs, metas = [], []
    for cid in fused_ids:
        if cid in dense_map:
            doc, meta = dense_map[cid]
        elif cid in fetched:
            doc, meta = fetched[cid]
        else:
            continue
        docs.append(doc)
        metas.append(meta)
    return docs, metas


def search_reference_data(query: str, top_k: int = 3) -> str:
    try:
        count = get_reference_collection().count()
    except Exception:
        return ""

    if count == 0:
        return ""

    results = get_reference_collection().query(
        query_texts=[query],
        n_results=min(top_k, count),
        include=["documents", "metadatas"]
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if HYBRID_SEARCH_ENABLED and documents:
        try:
            documents, metadatas = _hybrid_fuse_chunks(
                "reference", query, top_k, results.get("ids", [[]])[0],
                documents, metadatas, predicates=None)
        except Exception as e:
            print(f"[rag] hybrid fusion failed (reference), dense-only: {e}")

    # Post-RRF Laya Typed-Decision Filtering: drop chunks classified as irrelevant
    if LAYA_ENABLED and is_laya_available() and documents:
        try:
            cand_chunks = [{"text": d, "metadata": m} for d, m in zip(documents, metadatas)]
            kept, diag = rerank_and_filter_chunks(query, cand_chunks, max_chars=MAX_REFERENCE_RESULT_CHARS)
            if not kept:
                return ""
            documents = [c["text"] for c in kept]
            metadatas = [c["metadata"] for c in kept]
        except Exception as e:
            print(f"[rag] Laya chunk filtering failed (reference): {e}")

    if not documents:
        return ""

    tagged = [f"[{build_reference_excerpt_label(meta)}]\n{doc}" for doc, meta in zip(documents, metadatas)]
    combined = "\n---\n".join(tagged)
    if len(combined) > MAX_REFERENCE_RESULT_CHARS:
        combined = combined[:MAX_REFERENCE_RESULT_CHARS] + "\n[...reference excerpt truncated...]"
    return combined


def build_excerpt_label(meta: dict) -> str:
    meta = meta or {}
    source = meta.get("source", "unknown")
    person = meta.get("person") or (extract_person_key(source) if source != "unknown" else "")
    date_range = meta.get("date_range")
    source_type = meta.get("source_type")

    stem = Path(source).stem
    stem_lower = stem.lower()
    is_note = any(stem_lower.startswith(p) for p in NOTE_FILE_PREFIXES)

    if source_type == "bot_conversation":
        label = "Past conversation with you (bot memory)"
    elif is_note or not person or source_type == "notes":
        title = stem.replace("_", " ").strip().title()
        label = f"Personal Note: {title}"
    else:
        label = f"Chat with {person.capitalize()}"

    if date_range:
        label += f" | {date_range}"

    return label


def expand_bot_conversation_matches(documents: list, metadatas: list) -> list[tuple[str, dict]]:
    seen_sources = set()
    expanded = []

    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        if meta.get("source_type") != "bot_conversation":
            expanded.append((doc, meta))
            continue

        source = meta.get("source")
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)

        file_path = PERSONAL_DATA_DIR / source
        try:
            try:
                full_text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                full_text = file_path.read_text(encoding="latin1")
        except OSError:
            full_text = doc

        if len(full_text) > MAX_EXPANDED_SESSION_CHARS:
            full_text = full_text[:MAX_EXPANDED_SESSION_CHARS] + "\n[...session excerpt truncated...]"

        expanded.append((full_text, meta))

    return expanded


def detect_mentioned_person(query: str, known_people: set[str]) -> str | None:
    if not query or not known_people:
        return None
    # Sort candidates by length descending so longer names (e.g. "ravindra") take precedence over substrings ("ravi")
    candidates = sorted([p.strip() for p in known_people if p and p.strip()], key=lambda x: len(x), reverse=True)
    matches = []
    for person in candidates:
        m = re.search(rf"\b{re.escape(person)}\b", query, re.IGNORECASE)
        if m:
            matches.append((m.start(), len(person), person))
    if not matches:
        return None
    # Sort by earliest position in query, then longest match
    matches.sort(key=lambda x: (x[0], -x[1]))
    return matches[0][2]


KNOWN_PEOPLE_PAGE_SIZE = 1000  # ChromaDB's SQLite backend errors on unpaginated
                                # .get() calls once a collection reaches roughly
                                # tens of thousands of rows ("too many SQL
                                # variables"). This collection is well past that
                                # (60k+ chunks), so get_known_people() must page
                                # through results instead of fetching all at once.


_known_people_cache = {"count": -1, "people": set(), "expires_at": 0.0}
KNOWN_PEOPLE_TTL_SECONDS = 300.0  # 5 minutes TTL


def invalidate_known_people_cache():
    """Invalidates the in-memory cache of known people so subsequent calls refresh from ChromaDB."""
    global _known_people_cache
    _known_people_cache = {"count": -1, "people": set(), "expires_at": 0.0}


def get_known_people() -> set[str]:
    """Returns the set of unique person names mentioned across indexed data,
    cached with a 300-second TTL to avoid repeated collection.count() queries."""
    global _known_people_cache
    now = time.monotonic()
    if now < _known_people_cache.get("expires_at", 0.0) and _known_people_cache.get("people"):
        return _known_people_cache["people"]

    try:
        collection = get_collection()
        count = collection.count()
        if count == 0:
            return set()

        if _known_people_cache.get("count") == count and _known_people_cache.get("people"):
            _known_people_cache["expires_at"] = now + KNOWN_PEOPLE_TTL_SECONDS
            return _known_people_cache["people"]

        people = set()
        offset = 0
        while offset < count:
            page = collection.get(
                include=["metadatas"], limit=KNOWN_PEOPLE_PAGE_SIZE, offset=offset
            )
            page_metas = page.get("metadatas", [])
            if not page_metas:
                break
            for meta in page_metas:
                if meta and meta.get("person"):
                    people.add(meta["person"])
            offset += KNOWN_PEOPLE_PAGE_SIZE
        _known_people_cache = {
            "count": count,
            "people": people,
            "expires_at": now + KNOWN_PEOPLE_TTL_SECONDS
        }
        return people
    except Exception as e:
        # Fail-open, but never silently: an empty set here is
        # indistinguishable from "no known people" downstream and quietly
        # degrades person-tagging + retrieval. Make the cause visible.
        print(f"[rag] get_known_people() failed, returning empty set: {e}")
        return set()


TRIVIAL_MESSAGE_PATTERNS = {
    "hi", "hey", "hello", "yo", "sup", "hii", "hiii", "heyy", "heyyy",
    "lol", "lmao", "rofl", "ok", "okay", "k", "kk", "cool", "nice",
    "thanks", "thank you", "ty", "thx", "bye", "gn", "gm", "good morning",
    "good night", "wassup", "whats up", "what's up", "yeah", "yep", "nope",
    "sure", "alright"
}


def is_trivial_message(message: str) -> bool:
    cleaned = message.strip().lower().rstrip("!?. ")
    if cleaned in TRIVIAL_MESSAGE_PATTERNS:
        return True
    # Compound greetings (e.g. "hey good morning", "hi hello")
    for pat in ("good morning", "good night", "what's up", "whats up", "thank you"):
        if pat in cleaned:
            remainder = cleaned.replace(pat, "").strip(" ,!?.")
            if not remainder or remainder in TRIVIAL_MESSAGE_PATTERNS:
                return True
    return False


import calendar

_last_laya_filtered_count: int = 0


def get_last_laya_filtered_count() -> int:
    global _last_laya_filtered_count
    return _last_laya_filtered_count


VAGUE_SCHEDULING_PATTERNS = [
    "what time are we", "what time did we agree", "when are we meeting",
    "are we meeting", "what time we're meeting", "when are we hanging out",
    "what are we doing", "what time did we say we're meeting", "what time we said we're meeting"
]

META_MEMORY_PATTERNS = [
    "our conversation", "our conversations", "our chat", "our chats",
    "what did we talk about", "what did we discuss", "have we talked about",
    "did i tell you", "did i mention", "have i told you", "have i mentioned",
    "do you remember what we", "do you remember when we", "remember what we talked",
    "remember when we talked"
]


def is_vague_scheduling_question(message: str) -> bool:
    msg = message.lower()
    return any(p in msg for p in VAGUE_SCHEDULING_PATTERNS)


def is_meta_memory_question(message: str) -> bool:
    msg = message.lower()
    return any(p in msg for p in META_MEMORY_PATTERNS)


def should_bypass_personal_retrieval(query: str, laya_routing: dict | None = None, known_people: set[str] | None = None) -> bool:
    """
    Checks whether personal RAG retrieval should be bypassed:
    1. Standard trivial messages ("ok", "lol", "hey, good morning", etc.).
    2. Vague scheduling questions with the bot ("what time are we meeting?") where no person is mentioned.
    3. Laya classified as 'no_personal_context_needed' with confidence (>= 0.60)
       AND no known person or date reference is detected in the query.
    """
    if is_trivial_message(query):
        return True

    # Vague scheduling question without a mentioned person bypasses external chats
    if is_vague_scheduling_question(query):
        if known_people is None:
            try:
                known_people = get_known_people()
            except Exception:
                known_people = set()
        if detect_mentioned_person(query, known_people) is None:
            return True

    query_lower = query.lower()
    # Casual check-ins and meta-conversation about the chat itself
    casual_or_meta_phrases = [
        "how r u", "how are you", "how you doing", "hows it going", "how's it going",
        "u just started", "you just started", "going crazy", "werent there", "weren't there",
        "needed help", "trying to share", "share my difficulties",
    ]
    if any(p in query_lower for p in casual_or_meta_phrases):
        if known_people is None:
            try:
                known_people = get_known_people()
            except Exception:
                known_people = set()
        if detect_mentioned_person(query, known_people) is None:
            return True

    if laya_routing and not laya_routing.get("used_fallback"):
        ret_dec = laya_routing.get("retrieval_decision")
        ret_conf = laya_routing.get("retrieval_confidence", 0.0)
        if ret_dec == "no_personal_context_needed" and ret_conf >= 0.60:
            if known_people is None:
                try:
                    known_people = get_known_people()
                except Exception:
                    known_people = set()
            if detect_mentioned_person(query, known_people) is None and parse_date_reference(query) is None:
                return True

    return False


WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH_NAMES_NO_MAY = [
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
]
MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def parse_date_reference(message: str) -> tuple[date, date] | None:
    msg = message.lower().strip()
    today = date.today()

    # Guard against casual conversational phrases using "today" or conversational "yesterday"
    casual_date_patterns = [
        r"\bhow\s+(?:r|are)\s+u\s+today\b",
        r"\bhow\s+(?:r|are)\s+you\s+today\b",
        r"\bhow\s+is\s+it\s+going\s+today\b",
        r"\bhow\s+was\s+your\s+day\b",
        r"\bwhat'?s\s+up\s+today\b",
        r"\byesterday\s+(?:when\s+)?(?:i\s+tried|i\s+was|you|u\s+just|u\s+started|we\s+talked|you\s+went|u\s+went)\b",
        r"\bu\s+(?:weren'?t|were\s+not|went|just\s+started)\b",
    ]
    if any(re.search(p, msg) for p in casual_date_patterns):
        return None

    if "day before yesterday" in msg:
        d = today - timedelta(days=2)
        return d, d
    if "yesterday" in msg:
        d = today - timedelta(days=1)
        return d, d
    if re.search(r"\b(?:earlier\s+today|from\s+today|chats?\s+today|today'?s\s+chat|events?\s+today)\b", msg) or msg == "today":
        return today, today
    if "last week" in msg:
        this_week_start = today - timedelta(days=today.weekday())
        last_week_start = this_week_start - timedelta(days=7)
        return last_week_start, this_week_start - timedelta(days=1)
    if "this week" in msg:
        return today - timedelta(days=today.weekday()), today

    for i, weekday_name in enumerate(WEEKDAY_NAMES):
        if f"last {weekday_name}" in msg:
            days_since = (today.weekday() - i) % 7
            days_since = days_since or 7  
            d = today - timedelta(days=days_since)
            return d, d

    # Guard against lowercase auxiliary verb "may" (e.g. "I may be late", "may I ask")
    is_may_month = bool(
        re.search(r"\b(?:in\s+)?May\s+\d{1,4}\b", message)
        or re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?May\b", message)
        or re.search(r"\b(?:in|during)\s+May\b", message)
        or re.search(r"\bMay\s+(?:19|20)\d{2}\b", message, re.IGNORECASE)
    )

    has_other_month = any(re.search(rf"\b{m}\b", msg) for m in MONTH_NAMES_NO_MAY)
    has_month = has_other_month or is_may_month

    # Require explicit date separators (/ or . or full YYYY-MM-DD); avoid duration ranges like "1-2 hours" or "3-4 days"
    has_date_format = bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", msg)) or bool(re.search(r"\b\d{1,2}[/.]\d{1,2}(?:[/.]\d{2,4})?\b", msg))
    has_year = bool(re.search(r"\b(?:in|year|since)\s+(?:19|20)\d{2}\b", msg))

    # Month-only match (e.g. "in march", "during april", "march 2024"): expand to entire month range
    if has_month and not has_date_format:
        has_day = bool(re.search(r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[a-z]+|[a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\b", msg))
        if not has_day:
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", msg)
            year = int(year_match.group(1)) if year_match else today.year
            for m_name, m_num in MONTH_MAP.items():
                if m_name == "may" and not is_may_month:
                    continue
                if re.search(rf"\b{m_name}\b", msg):
                    last_day = calendar.monthrange(year, m_num)[1]
                    return date(year, m_num, 1), date(year, m_num, last_day)

    if has_month or has_date_format or has_year:
        try:
            parsed = dateutil_parser.parse(message, fuzzy=True, dayfirst=True, default=datetime(today.year, 1, 1))
            return parsed.date(), parsed.date()
        except (ValueError, OverflowError):
            pass

    return None


def search_personal_data(query: str, top_k: int = 6, max_result_chars: int = MAX_PERSONAL_RESULT_CHARS) -> str:
    try:
        count = get_collection().count()
    except Exception as e:
        # Fail-open, but never silently: "" reads downstream as "nothing
        # found" rather than "retrieval is broken" — the model then answers
        # confidently with no context. Log so the failure is diagnosable.
        print(f"[rag] search_personal_data(): collection unavailable, "
              f"returning no context: {e}")
        return ""

    if count == 0:
        return ""

    known_people = get_known_people()
    mentioned_person = detect_mentioned_person(query, known_people)

    if is_meta_memory_question(query) and not mentioned_person:
        results = get_collection().query(
            query_texts=[query],
            n_results=min(top_k, count),
            where={"source_type": "bot_conversation"},
            include=["documents", "metadatas", "distances"]
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return ""

        if HYBRID_SEARCH_ENABLED:
            try:
                documents, metadatas = _hybrid_fuse_chunks(
                    "personal", query, top_k, results.get("ids", [[]])[0],
                    documents, metadatas,
                    predicates=[_bm25_predicate_bot_only])
            except Exception as e:
                print(f"[rag] hybrid fusion failed (meta-memory), dense-only: {e}")

        # Post-RRF Laya filtering for bot conversation matches
        if LAYA_ENABLED and is_laya_available() and documents:
            try:
                cand_chunks = [{"text": d, "metadata": m} for d, m in zip(documents, metadatas)]
                kept, diag = rerank_and_filter_chunks(query, cand_chunks, max_chars=max_result_chars)
                if not kept:
                    return ""
                documents = [c["text"] for c in kept]
                metadatas = [c["metadata"] for c in kept]
            except Exception as e:
                print(f"[rag] Laya chunk filtering failed (meta-memory): {e}")

        expanded = expand_bot_conversation_matches(documents, metadatas)
        tagged = [f"[{build_excerpt_label(meta)}]\n{text}" for text, meta in expanded]
        combined = "\n---\n".join(tagged)
        if len(combined) > max_result_chars:
            combined = combined[:max_result_chars] + "\n[...personal context truncated...]"
        return combined

    date_range = parse_date_reference(query)

    where_conditions = [{"source_type": {"$ne": "bot_conversation"}}]
    if mentioned_person:
        where_conditions.append({"person": mentioned_person})
    if date_range:
        start_epoch = _epoch_for(date_range[0])
        end_epoch = _epoch_for(date_range[1]) + 86400  
        where_conditions.append({"date_start_epoch": {"$lte": end_epoch}})
        where_conditions.append({"date_end_epoch": {"$gte": start_epoch}})

    query_kwargs = {
        "query_texts": [query],
        "n_results": min(top_k, count),
        "include": ["documents", "metadatas", "distances"],
        "where": where_conditions[0] if len(where_conditions) == 1 else {"$and": where_conditions}
    }

    results = get_collection().query(**query_kwargs)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    # No silent unfiltered fallback on date miss:
    # If the user explicitly asked for a specific date and date-filtered retrieval returns 0 hits,
    # return empty context with an explicit note instead of silently falling back to unrelated dates.
    if date_range and not documents:
        return "No records found for that date."

    if not documents:
        return ""

    if distances:
        best_distance = min(distances)
        margin = max(best_distance * 0.5, 0.15)
        threshold = best_distance + margin
        filtered = [
            (id_, doc, meta) for id_, doc, meta, dist in zip(ids, documents, metadatas, distances)
            if dist <= threshold
        ]
        if filtered:
            ids, documents, metadatas = zip(*filtered)
            ids = list(ids)
            documents = list(documents)
            metadatas = list(metadatas)

    if HYBRID_SEARCH_ENABLED and ids:
        try:
            documents, metadatas = _hybrid_fuse_chunks(
                "personal", query, top_k, ids, documents, metadatas,
                predicates=_bm25_predicate_personal(mentioned_person, date_range))
        except Exception as e:
            print(f"[rag] hybrid fusion failed (personal), dense-only: {e}")

    # Post-RRF Laya Typed-Decision Filtering: drop irrelevant chunks before assembly
    global _last_laya_filtered_count
    _last_laya_filtered_count = 0
    if LAYA_ENABLED and is_laya_available() and documents:
        try:
            cand_chunks = [{"text": d, "metadata": m} for d, m in zip(documents, metadatas)]
            kept, diag = rerank_and_filter_chunks(query, cand_chunks, max_chars=max_result_chars)
            _last_laya_filtered_count = len(cand_chunks) - len(kept)
            if not kept:
                return ""
            documents = [c["text"] for c in kept]
            metadatas = [c["metadata"] for c in kept]
        except Exception as e:
            print(f"[rag] Laya chunk filtering failed (personal): {e}")

    tagged = []
    for doc, meta in zip(documents, metadatas):
        tagged.append(f"[{build_excerpt_label(meta)}]\n{doc}")

    combined = "\n---\n".join(tagged)
    if len(combined) > max_result_chars:
        combined = combined[:max_result_chars] + "\n[...personal context truncated...]"
    return combined


if __name__ == "__main__":
    index_personal_data()
    index_reference_data()