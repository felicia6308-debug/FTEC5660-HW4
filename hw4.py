"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

STOPWORDS = frozenset("""
a an the of in on at for to from by with and or as its it this that these those
what which who whom whose how much many did do does was were is are be been being
had has have will would can could should shall may might
give gives gave given set sets setting report reports reported reporting
generate generated generates earn earned earning
hold held holding announce announced announces
audit audits audited auditing spend spent spending
guide guided guides guidance forecast forecasts forecasted
company companies usd million millions percent percents
fiscal fy year years end ending ended
fourth quarter quarterly
value number numbers
city headquartered firm
""".split())

NAME_BLOCK = frozenset("""
chief executive officer financial operating technology director chair chairman head
audit auditor independent board rating target price revenue total gross net value end
fiscal year fourth quarter call review summary investment share shares equity income
cash assets liabilities profit segment results estimate estimates
""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.\-']*")
YEAR_RE = re.compile(r"(?<!\d)(20\d\d)(?!\d)")
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")
NUMERIC_HINT = re.compile(r"how\s+much|how\s+many|in\s+usd\s+million|in\s+percent|what\s+was\s+the\s+value", re.I)
TABLE_BOOST = 0.15

KNOWN_COMPANIES: dict[str, str] = {}
BROKER_NAMES: set[str] = set()
CORPUS_VOCAB: set[str] = set()
COMPANY_WORDS: set[str] = set()


def _squash(text) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def _span(value) -> int:
    try:
        return max(1, int(str(value).strip()))
    except (TypeError, ValueError):
        return 1


def _table_rows(table):
    """Cell texts per row, with rowspan and colspan expanded into a rectangular grid."""
    grid: dict[tuple[int, int], str] = {}
    rows = []
    width = 0
    for r, tr in enumerate(table.find_all("tr")):
        row = []
        col = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (r, col) in grid:
                col += 1
            text = _squash(cell.get_text(" "))
            columns, down = _span(cell.get("colspan")), _span(cell.get("rowspan"))
            for dr in range(down):
                for dc in range(columns):
                    grid[(r + dr, col + dc)] = text
            row.append((text, columns, down))
            col += columns
        width = max(width, col)
        rows.append(row)
    return rows, [[grid.get((r, c), "") for c in range(width)] for r in range(len(rows))], width


def table_to_text(table_html: str) -> str:
    """Task 1.1: one line per table row, every value with its row label and its column header."""
    soup = BeautifulSoup(str(table_html), "lxml")
    table = soup.find("table")
    if table is None:
        return _squash(table_html)

    caption = ""
    cap = table.find("caption")
    if cap is not None:
        caption = _squash(cap.get_text(" "))

    raw, grid, width = _table_rows(table)
    lines = [caption] if caption else []
    if len(grid) < 2:
        return "\n".join(lines)

    headers = grid[0][1:]
    for r in range(1, len(grid)):
        row = [_squash(cell) for cell in grid[r]]
        cells = raw[r]
        if len(cells) == 1 and cells[0][1] >= width:
            if row[0]:
                lines.append(row[0])
            continue
        label = row[0]
        pairs = [(headers[i], row[i + 1]) for i in range(len(headers))
                 if i + 1 < len(row) and row[i + 1] and headers[i]]
        if not label or not pairs:
            continue
        line = " | ".join([label] + [f"{h}: {v}" for h, v in pairs])
        if len(line) <= MAX_LINE_CHARS:
            lines.append(line)
        else:
            lines.extend(f"{label} | {h}: {v}" for h, v in pairs)
    return "\n".join(lines)


def _document_meta(html: str):
    """(company, doc_type, fiscal_year) read from the first lines of the document."""
    soup = BeautifulSoup(html, "lxml")
    text = _squash(soup.get_text(" "))
    h1 = soup.find("h1")
    company = _squash(h1.get_text(" ")) if h1 is not None else ""
    company = re.sub(r"\s*\([^()]*\)\s*$", "", company).strip() or company

    if re.search(r"earnings\s+(call|conference)", text, re.I):
        doc_type = "earnings_call"
    elif re.search(r"results\s+review", text, re.I) or soup.select_one(".broker") is not None:
        doc_type = "research_report"
    else:
        doc_type = "annual_report"

    if doc_type == "annual_report":
        match = re.search(r"Annual\s+Report\s+(20\d\d)", text, re.I)
    elif doc_type == "research_report":
        match = re.search(r"FY\s*(20\d\d)\s+results\s+review", text, re.I)
    else:
        match = re.search(r"Full\s+Year\s+FY?\s*(20\d\d)", text, re.I)
    if match is None:
        match = YEAR_RE.search(text)
    return company, doc_type, int(match.group(1)) if match else 2025


def _remember(company: str, soup) -> None:
    if company:
        KNOWN_COMPANIES[company_key(company)] = company
        COMPANY_WORDS.update(company_key(company).split())
    broker = soup.select_one(".broker")
    if broker is not None:
        name = _squash(broker.get_text(" ")).split("|")[0].strip()
        if name:
            BROKER_NAMES.add(company_key(name))


def _blocks(soup):
    """(section title, text) pairs in document order, with tables already flattened."""
    out = []
    section, subsection = "", ""
    for el in soup.find_all(["h2", "h3", "p", "li", "table"]):
        if el.find_parent("nav") is not None or el.find_parent("table") is not None:
            continue
        if el.name == "li" and el.find_parent("li") is not None:
            continue
        if el.name == "p" and el.find_parent(["li", "p"]) is not None:
            continue
        if el.name == "h2":
            section, subsection = _squash(el.get_text(" ")), ""
            continue
        if el.name == "h3":
            subsection = _squash(el.get_text(" "))
            continue
        if section.lower() == "contents":
            continue
        text = table_to_text(str(el)) if el.name == "table" else _squash(el.get_text(" "))
        if not text:
            continue
        out.append((f"{section} - {subsection}" if subsection else section, text))
    return out


def _split(text: str, budget: int):
    """Pieces of at most `budget` characters, never inside a table line."""
    if len(text) <= budget:
        return [text]
    sep = "\n" if "\n" in text else " "
    parts = text.split(sep)
    caption = parts[0] if sep == "\n" and len(parts) > 1 and len(parts[0]) < budget // 3 else ""
    room = max(1, budget - (len(caption) + 1 if caption else 0))
    pieces, current, size = [], [], 0
    for part in parts:
        if current and size + len(part) + 1 > room:
            pieces.append(sep.join(current))
            current, size = [], 0
        while len(part) > room:
            if current:
                pieces.append(sep.join(current))
                current, size = [], 0
            pieces.append(part[:room])
            part = part[room:]
        current.append(part)
        size += len(part) + 1
    if current:
        pieces.append(sep.join(current))
    out = []
    for i, piece in enumerate(pieces):
        if caption and i and caption not in piece:
            piece = caption + sep + piece
        out.append(piece)
    return [p for p in out if p]


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: section-aware chunks with metadata and a short contextual header."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    company, doc_type, fiscal_year = _document_meta(html)
    _remember(company, soup)

    metadata = {"source": file_name, "company": company,
                "fiscal_year": fiscal_year, "doc_type": doc_type}
    budget = max(200, MAX_CHUNK_CHARS - 60)

    chunks: list[Document] = []
    current, current_title, size = [], None, 0

    def flush() -> None:
        nonlocal current, size, current_title
        if not current:
            return
        head = f"Section: {current_title}" if current_title else ""
        text = (head + "\n" if head else "") + "\n\n".join(current)
        for piece in _split(text, MAX_CHUNK_CHARS):
            chunks.append(Document(page_content=piece, metadata=dict(metadata)))
        current, size = [], 0

    for title, block in _blocks(soup):
        CORPUS_VOCAB.update(_bm25_tokens(block))
        for piece in _split(block, budget):
            if current and (title != current_title or size + len(piece) + 2 > budget):
                flush()
            current_title = title
            current.append(piece)
            size += len(piece) + 2
    flush()
    return chunks


def _bm25_tokens(text: str) -> list[str]:
    return [_stem(token) for token in TOKEN_RE.findall(str(text).lower())]


def _table_ratio(text: str) -> float:
    """Share of the chunk's lines that look like a flattened table row."""
    lines = text.split("\n")
    return sum(1 for line in lines if " | " in line) / len(lines) if lines else 0.0


def _query_tokens(question: str) -> list[str]:
    """Query terms with the question words and the company names removed."""
    tokens = [t for t in _bm25_tokens(question)
              if t not in STOPWORDS and t not in COMPANY_WORDS]
    return tokens or _bm25_tokens(question)


def _stem(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "sh", "ch", "x", "z", "s'")):
        return word[:-1]
    return word


def _title_case(name: str) -> str:
    return " ".join(word[:1].upper() + word[1:] for word in name.split())


def _companies_in(question: str):
    """Every company the question names, matched from the corpus names so decoys are kept."""
    entries = sorted(((name, company_key(name)) for name in KNOWN_COMPANIES.values()),
                     key=lambda entry: -len(entry[1].split()))
    heads = {key.split(" ")[0] for _, key in entries}
    words = [match.group(0) for match in WORD_RE.finditer(question)]
    found: list[str] = []

    def add(name: str) -> None:
        key = company_key(name)
        if not key or key in BROKER_NAMES:
            return
        if all(company_key(other) != key for other in found):
            found.append(name)

    def bare(token: str) -> str:
        return re.sub(r"'s?$", "", token).lower()

    i = 0
    while i < len(words):
        if bare(words[i]) in heads:
            hit = None
            for name, key in entries:
                parts = key.split(" ")
                if i + len(parts) <= len(words) and all(
                        bare(words[i + k]) == parts[k] for k in range(len(parts))):
                    hit = (name, len(parts))
                    break
            if hit is not None:
                add(hit[0])
                i += hit[1]
                continue
            parts, j = [], i
            while j < len(words):
                token = words[j]
                if j > i and (not token[:1].isupper() or bare(token) in NAME_BLOCK
                              or bare(token) in heads):
                    break
                parts.append(token)
                j += 1
                if token.endswith("'") or token.endswith("'s"):
                    break
            add(_squash(re.sub(r"'s?$", "", " ".join(parts))))
            i = j
            continue
        i += 1
    # Decoys may have no words in common with indexed companies.
    if found and not re.search(r"\b(compare|between|versus|vs|than|which|and)\b", question, re.I):
        return found
    for match in re.finditer(r"\b[A-Z][a-zA-Z&.-]+(?:[ \t]+[A-Z][a-zA-Z&.-]+)+", question):
        parts = match.group().split()
        while parts and (parts[0].lower() in STOPWORDS or parts[0].lower() == "compare"):
            parts.pop(0)
        while parts and parts[-1].lower() in NAME_BLOCK:
            parts.pop()
        if len(parts) < 2:
            continue
        name = " ".join(parts)
        key = company_key(name)
        if (key in BROKER_NAMES or any(part.lower() in NAME_BLOCK for part in parts)
                or re.search(r"\b(research|capital|securities|llp|december|january|asia|pacific|middle east|africa|europe)\b", key)
                or any(company_key(other) in key or key in company_key(other) for other in found)):
            continue
        add(name)
    return found


def parse_query(question: str) -> dict:
    """Task 2.2: the companies, the fiscal year and the document type the question names."""
    text = _squash(question).replace("’", "\x27")
    companies = _companies_in(text)

    match = YEAR_RE.search(text)
    fiscal_year = int(match.group(1)) if match else None

    if re.search(r"earnings\s+call", text, re.I):
        doc_type = "earnings_call"
    elif re.search(r"\b(rating|target\s+price|forecast|forecasts|results\s+review|estimates?|broker)\b",
                   text, re.I):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"

    return {"companies": companies, "fiscal_year": fiscal_year, "doc_type": doc_type}


def _doc_years(doc_type: str, doc_year):
    """The fiscal years whose figures a document of this type prints."""
    year = as_year(doc_year)
    if year is None:
        return ()
    if doc_type == "annual_report":
        return (year, year - 1)
    if doc_type == "earnings_call":
        return (year, year + 1)
    return (year, year + 1, year + 2)


def build_index(chunks: list[Document]):
    """Task 2: a BM25 index over every chunk, with the position of each chunk."""
    tokens = [_bm25_tokens(c.page_content) for c in chunks]
    for token in tokens:
        CORPUS_VOCAB.update(token)
    return {"chunks": list(chunks), "bm25": BM25Okapi(tokens) if tokens else None,
            "position": {id(c): i for i, c in enumerate(chunks)}}


def _answerable(question: str, chunks) -> bool:
    """False when the question asks for something the corpus never mentions."""
    if not chunks or not CORPUS_VOCAB:
        return bool(chunks)
    text = re.sub(r"'s\b", "", _squash(question)).replace("'", " ")
    for word in TOKEN_RE.findall(text.lower()):
        if word in STOPWORDS or len(word) < 3 or any(ch.isdigit() for ch in word):
            continue
        if word in COMPANY_WORDS:
            continue
        if _stem(word) not in CORPUS_VOCAB:
            return False
    return True


def retrieve(index, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED chunks, or [] when the corpus cannot answer."""
    chunks = index.get("chunks") or []
    if not chunks:
        return []

    query = parse_query(question)
    wanted = [company_key(name) for name in query.get("companies") or []]
    known = [key for key in wanted if key in KNOWN_COMPANIES]
    if len(known) != len(wanted):
        return []
    doc_type = query.get("doc_type") or "annual_report"
    year = query.get("fiscal_year")

    pool = [c for c in chunks
            if (c.metadata or {}).get("doc_type") == doc_type
            and (not known or company_key((c.metadata or {}).get("company")) in known)]
    if not pool:
        return []

    if year is not None:
        covered = [c for c in pool
                   if year in _doc_years(doc_type, (c.metadata or {}).get("fiscal_year"))]
        if not covered:
            return []
        exact = [c for c in covered
                 if as_year((c.metadata or {}).get("fiscal_year")) == year]
        pool = exact or covered

    scores = index["bm25"].get_scores(_query_tokens(question)) if index.get("bm25") else None
    position = index["position"]

    boost = TABLE_BOOST if NUMERIC_HINT.search(question) else 0.0

    def weight(chunk: Document) -> float:
        value = scores[position[id(chunk)]] if scores is not None else 0.0
        return value * (1.0 + boost * _table_ratio(chunk.page_content)) if boost else value

    ranked = sorted(pool, key=weight, reverse=True)
    groups = known or [None]
    by_company = {key: [c for c in ranked
                        if company_key((c.metadata or {}).get("company")) == key]
                  for key in known}

    picked, seen = [], set()
    base, extra = divmod(MAX_RETRIEVED, len(groups))
    for i, key in enumerate(groups):
        quota = base + (1 if i < extra else 0)
        for chunk in (by_company.get(key, []) if key else ranked):
            if quota <= 0:
                break
            if id(chunk) not in seen:
                picked.append(chunk)
                seen.add(id(chunk))
                quota -= 1
    for chunk in ranked:
        if len(picked) >= MAX_RETRIEVED:
            break
        if id(chunk) not in seen:
            picked.append(chunk)
            seen.add(id(chunk))

    picked = picked[:MAX_RETRIEVED]
    return picked if _answerable(question, picked) else []


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
