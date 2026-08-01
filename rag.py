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
from pathlib import Path
from datetime import date, datetime, timedelta
from dateutil import parser as dateutil_parser
import chromadb
from chromadb.utils import embedding_functions

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
# "12/06/2026, 14:32 - Rahul: message text"
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

client = chromadb.PersistentClient(path=str(CHROMA_STORE_DIR))

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
        _collection = client.get_or_create_collection(
            name="personal_data",
            embedding_function=get_embedding_fn()
        )
    return _collection

def get_reference_collection():
    global _reference_collection
    if _reference_collection is None:
        _reference_collection = client.get_or_create_collection(
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


def extract_person_key(filename: str) -> str:
    name = re.sub(r"\.txt$", "", filename, flags=re.IGNORECASE)
    name = re.sub(r"\d+.*$", "", name) 
    return name.lower()


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
    for batch_start in range(0, len(chunks), MAX_BATCH):
        batch_end = batch_start + MAX_BATCH
        get_collection().upsert(
            ids=ids[batch_start:batch_end],
            documents=chunks[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end]
        )

    print(f"Indexed: {relative_source} ({len(chunks)} chunks, source_type={source_type})")
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


def _chunk_reference_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        chunks.extend(_split_oversized_paragraph(p))
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

        chunks = _chunk_reference_text(text)
        if not chunks:
            continue

        relative_source = str(file_path.relative_to(REFERENCE_DATA_DIR))
        safe_id_base = re.sub(r"[^a-zA-Z0-9]", "_", relative_source)

        try:
            get_reference_collection().delete(where={"source": relative_source})
        except Exception:
            pass  

        ids = [f"ref_{safe_id_base}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": relative_source} for _ in chunks]

        get_reference_collection().upsert(ids=ids, documents=chunks, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"Indexed reference: {relative_source} ({len(chunks)} chunks)")

    print(f"\nReference data done. {len(txt_files)} file(s), {total_chunks} chunks.")


def search_reference_data(query: str, top_k: int = 3) -> str:
    try:
        count = get_reference_collection().count()
    except Exception:
        return ""

    if count == 0:
        return ""

    results = get_reference_collection().query(query_texts=[query], n_results=min(top_k, count))
    documents = results.get("documents", [[]])[0]

    if not documents:
        return ""

    combined = "\n---\n".join(documents)
    if len(combined) > MAX_REFERENCE_RESULT_CHARS:
        combined = combined[:MAX_REFERENCE_RESULT_CHARS] + "\n[...reference excerpt truncated...]"
    return combined


def build_excerpt_label(meta: dict) -> str:
    meta = meta or {}
    person = meta.get("person")
    date_range = meta.get("date_range")
    source = meta.get("source", "unknown")
    source_type = meta.get("source_type")

    if source_type == "bot_conversation":
        label = "Past conversation with you (bot memory)"
    elif person:
        label = f"Chat with {person.capitalize()}"
    else:
        label = f"Personal notes ({source})"

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
    query_lower = query.lower()
    for person in known_people:
        if person and person in query_lower:
            return person
    return None


def get_known_people() -> set[str]:
    try:
        all_data = get_collection().get(include=["metadatas"])
    except Exception:
        return set()

    people = set()
    for meta in all_data.get("metadatas", []):
        if meta and meta.get("person"):
            people.add(meta["person"])
    return people


TRIVIAL_MESSAGE_PATTERNS = {
    "hi", "hey", "hello", "yo", "sup", "hii", "hiii", "heyy",
    "lol", "lmao", "ok", "okay", "k", "kk", "cool", "nice",
    "thanks", "thank you", "ty", "bye", "gn", "gm", "good morning",
    "good night", "wassup", "whats up", "what's up"
}


def is_trivial_message(message: str) -> bool:
    cleaned = message.strip().lower().rstrip("!?. ")
    if cleaned in TRIVIAL_MESSAGE_PATTERNS:
        return True
    if len(cleaned) <= 4 and cleaned.isalpha():
        return True
    return False


META_MEMORY_PATTERNS = [
    "remember", "recall", "our conversation", "our conversations", "our chat",
    "our chats", "our past", "you and me", "you and i", "between us",
    "what did we talk", "what did we discuss", "have we talked",
    "have we discussed", "did i tell you", "did i mention",
    "have i told you", "have i mentioned"
]


def is_meta_memory_question(message: str) -> bool:
    msg = message.lower()
    return any(p in msg for p in META_MEMORY_PATTERNS)


WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
]


def parse_date_reference(message: str) -> tuple[date, date] | None:
    msg = message.lower()
    today = date.today()

    if "day before yesterday" in msg:
        d = today - timedelta(days=2)
        return d, d
    if "yesterday" in msg:
        d = today - timedelta(days=1)
        return d, d
    if "today" in msg:
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

    looks_date_like = bool(re.search(r"\d{1,2}(st|nd|rd|th)?[\s/-]", msg)) or any(m in msg for m in MONTH_NAMES)
    if looks_date_like:
        try:
            parsed = dateutil_parser.parse(message, fuzzy=True, dayfirst=True, default=datetime(today.year, 1, 1))
            return parsed.date(), parsed.date()
        except (ValueError, OverflowError):
            pass

    return None


def search_personal_data(query: str, top_k: int = 6, max_result_chars: int = MAX_PERSONAL_RESULT_CHARS) -> str:
    try:
        count = get_collection().count()
    except Exception:
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

    if not documents and date_range and mentioned_person:
        results = get_collection().query(
            query_texts=[query], n_results=min(top_k, count),
            where={"$and": [{"source_type": {"$ne": "bot_conversation"}}, {"person": mentioned_person}]},
            include=["documents", "metadatas", "distances"]
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

    if not documents:
        results = get_collection().query(
            query_texts=[query], n_results=min(top_k, count),
            where={"source_type": {"$ne": "bot_conversation"}},
            include=["documents", "metadatas", "distances"]
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

    if not documents:
        return ""

    if distances:
        best_distance = min(distances)
        margin = max(best_distance * 0.5, 0.15)
        threshold = best_distance + margin
        filtered = [
            (doc, meta) for doc, meta, dist in zip(documents, metadatas, distances)
            if dist <= threshold
        ]
        if filtered:
            documents, metadatas = zip(*filtered)

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