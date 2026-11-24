"""Task 5 ablation: start from B0 (naive RAG) and change one component per row.

    python3 ablation.py --data-folder public_test [--with-llm]

Each row is scored exactly like the runner in hw4.py. The Task 4 column needs a
DEEPSEEK_API_KEY; without one it is left empty.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup, NavigableString
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4

NUMERIC_HINT = re.compile(r"how\s+much|how\s+many|in\s+usd\s+million|in\s+percent|what\s+was\s+the\s+value", re.I)


# --------------------------------------------------------------------------- chunkers

def reset_state() -> None:
    """Forget the corpus-level caches so every variant sees only its own text."""
    hw4.KNOWN_COMPANIES.clear()
    hw4.BROKER_NAMES.clear()
    hw4.CORPUS_VOCAB.clear()
    hw4.COMPANY_WORDS.clear()


def naive_chunks(file_name: str, html: str, use_table_text: bool) -> list[Document]:
    """B0 / V1: BeautifulSoup text, RecursiveCharacterTextSplitter, no structure."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    company, doc_type, fiscal_year = hw4._document_meta(html)
    hw4._remember(company, soup)
    if use_table_text:
        for table in soup.find_all("table"):
            table.replace_with(NavigableString(hw4.table_to_text(str(table))))
    text = soup.get_text(" ")
    metadata = {"source": file_name, "company": company,
                "fiscal_year": fiscal_year, "doc_type": doc_type}
    splitter = RecursiveCharacterTextSplitter(chunk_size=hw4.MAX_CHUNK_CHARS, chunk_overlap=200)
    chunks = [Document(page_content=piece, metadata=dict(metadata))
              for piece in splitter.split_text(text)]
    for chunk in chunks:
        hw4.CORPUS_VOCAB.update(hw4._bm25_tokens(chunk.page_content))
    return chunks


def structured_chunks(file_name: str, html: str) -> list[Document]:
    """V2 onwards: the section-aware chunker of the final solution."""
    return hw4.chunk_document(file_name, html)


# --------------------------------------------------------------------------- retrievers

class Variant:
    """One row of the ablation table: a chunker plus a set of retrieval switches."""

    def __init__(self, name, chunker, use_table_text, metadata_filter, drop_stopwords,
                 table_boost, per_company, abstain, note, *, improved_tokens=False):
        self.name = name
        self.chunker = chunker
        self.use_table_text = use_table_text
        self.metadata_filter = metadata_filter
        self.drop_stopwords = drop_stopwords
        self.table_boost = table_boost
        self.per_company = per_company
        self.abstain = abstain
        self.note = note
        self.tokenize = hw4._bm25_tokens if improved_tokens else lambda text: text.lower().split()

    def chunks(self, corpus):
        reset_state()
        out = {}
        for name, html in corpus.items():
            out[name] = (self.chunker(name, html) if self.chunker is structured_chunks
                         else self.chunker(name, html, self.use_table_text))
        chunks = [c for name in sorted(out) for c in out[name]]
        all_chunks = [c for c in chunks if len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
        return out, all_chunks

    def index(self, all_chunks):
        tokens = [self.tokenize(c.page_content) for c in all_chunks]
        for token in tokens:
            hw4.CORPUS_VOCAB.update(token)
        return {"chunks": all_chunks, "bm25": BM25Okapi(tokens) if tokens else None,
                "position": {id(c): i for i, c in enumerate(all_chunks)}}

    @staticmethod
    def table_ratio(text):
        lines = text.split("\n")
        return sum(1 for line in lines if " | " in line) / len(lines) if lines else 0.0

    def retrieve(self, index, question):
        if self.abstain:
            return hw4.retrieve(index, question)
        chunks = index["chunks"]
        if not chunks:
            return []
        parsed = hw4.parse_query(question) if self.metadata_filter else {}
        wanted = [hw4.company_key(n) for n in parsed.get("companies") or []]
        known = [k for k in wanted if k in hw4.KNOWN_COMPANIES]
        doc_type = parsed.get("doc_type") or "annual_report"
        year = parsed.get("fiscal_year")

        pool = list(chunks)
        if self.metadata_filter:
            if len(known) != len(wanted):
                return []
            pool = [c for c in pool if c.metadata.get("doc_type") == doc_type
                    and (not known or hw4.company_key(c.metadata.get("company")) in known)]
            if not pool:
                return []
            if year is not None:
                covered = [c for c in pool if year in hw4._doc_years(doc_type, c.metadata.get("fiscal_year"))]
                if not covered:
                    return []
                exact = [c for c in covered if c.metadata.get("fiscal_year") == year]
                pool = exact or covered

        tokens = hw4._query_tokens(question) if self.drop_stopwords else self.tokenize(question)
        scores = index["bm25"].get_scores(tokens) if index["bm25"] else None
        position = index["position"]
        boost = 0.15 if (self.table_boost and NUMERIC_HINT.search(question)) else 0.0

        def weight(chunk):
            value = scores[position[id(chunk)]] if scores is not None else 0.0
            return value * (1.0 + boost * self.table_ratio(chunk.page_content)) if boost else value

        ranked = sorted(pool, key=weight, reverse=True)

        if self.per_company and known:
            groups = known
            base, extra = divmod(hw4.MAX_RETRIEVED, len(groups))
            picked, seen = [], set()
            by_company = {k: [c for c in ranked if hw4.company_key(c.metadata.get("company")) == k]
                          for k in known}
            for i, key in enumerate(groups):
                quota = base + (1 if i < extra else 0)
                for chunk in by_company.get(key, []):
                    if quota <= 0:
                        break
                    if id(chunk) not in seen:
                        picked.append(chunk)
                        seen.add(id(chunk))
                        quota -= 1
            for chunk in ranked:
                if len(picked) >= hw4.MAX_RETRIEVED:
                    break
                if id(chunk) not in seen:
                    picked.append(chunk)
                    seen.add(id(chunk))
            ranked = picked

        picked = ranked[:hw4.MAX_RETRIEVED]
        return picked


VARIANTS = [
    Variant("B0", naive_chunks, False, False, False, False, False, False,
            "naive RAG: get_text(), RecursiveCharacterTextSplitter(), BM25 top-5, never []"),
    Variant("V1", naive_chunks, True, False, False, False, False, False,
            "+ table_to_text replaces the number soup of get_text()"),
    Variant("V2", structured_chunks, True, False, False, False, False, False,
            "+ section-aware chunking with a section header"),
    Variant("V3", structured_chunks, True, True, False, False, False, False,
            "+ parse_query metadata filtering (company, year, doc type)"),
    Variant("V4", structured_chunks, True, True, False, False, False, False,
            "+ regex tokenization and light stemming for BM25", improved_tokens=True),
    Variant("V5", structured_chunks, True, True, True, False, False, False,
            "+ stop words and company names dropped from the BM25 query", improved_tokens=True),
    Variant("V6", structured_chunks, True, True, True, True, False, False,
            "+ table boost for questions that ask for a figure", improved_tokens=True),
    Variant("V7", structured_chunks, True, True, True, True, True, False,
            "+ one quota per company so a comparison gets both sides", improved_tokens=True),
    Variant("V8", structured_chunks, True, True, True, True, True, True,
            "+ lexical abstention for absent terms", improved_tokens=True),
]


# --------------------------------------------------------------------------- scoring

def score_variant(variant, corpus, questions, ground_truth, tables, docs_meta,
                  with_llm: bool, details: list) -> dict:
    chunks_by_file, all_chunks = variant.chunks(corpus)
    index = variant.index(all_chunks)
    row = {"variant": variant.name, "change": variant.note}

    if variant.use_table_text:
        scores = [hw4.table_score(hw4.table_to_text(t["html"]), t) for t in tables]
        row["1.1"] = hw4.POINTS["1.1"] * sum(scores) / len(scores)
    else:
        row["1.1"] = 0.0
    passed = sum(not hw4.document_problems(chunks_by_file.get(f), None, meta, f)
                 for f, meta in docs_meta.items()) / len(docs_meta)
    row["1.2"] = hw4.POINTS["1.2"] * passed

    parsed = {qid: (hw4.parse_query(q) if variant.metadata_filter else {})
              for qid, q in questions.items()}
    got = {qid: variant.retrieve(index, q) for qid, q in questions.items()}

    answerable = [q for q in questions if ground_truth[q]["evidence"]]
    unanswerable = [q for q in questions if not ground_truth[q]["evidence"]]
    valid = sum(isinstance(got[q], list) and 0 < len(got[q]) <= hw4.MAX_RETRIEVED for q in answerable)
    parse_mean = sum(hw4.parse_score(parsed[q], ground_truth[q]) for q in questions) / len(questions)
    recall = sum(hw4.retrieval_hit(got[q], ground_truth[q]) for q in answerable) / len(answerable)
    kept = sum(bool(got[q]) for q in answerable) / len(answerable)
    rejected = sum(not got[q] for q in unanswerable) / len(unanswerable)

    row["2.1"] = hw4.POINTS["2.1"] * valid / len(answerable)
    row["2.2"] = hw4.POINTS["2.2"] * parse_mean
    row["2.3"] = hw4.POINTS["2.3"] * hw4.clip((recall - hw4.RECALL_ZERO) / (hw4.RECALL_FULL - hw4.RECALL_ZERO))
    row["3"] = hw4.POINTS["3"] * hw4.clip(kept + rejected - 1)
    single = [q for q in answerable if len(ground_truth[q]["evidence"]) == 1]
    row["top1"] = (sum(hw4.retrieval_hit(got[q][:1], ground_truth[q]) for q in single)
                   / len(single)) if single else 0.0
    comparisons = [q for q in answerable if len(ground_truth[q]["evidence"]) > 1]
    both = 0
    for q in comparisons:
        want = {hw4.company_key(n) for n in ground_truth[q]["query"]["companies"]}
        have = {hw4.company_key(c.metadata.get("company")) for c in got[q]}
        both += want.issubset(have)
    row["both"] = both / len(comparisons) if comparisons else 0.0
    row["recall"] = recall
    row["kept"] = kept
    row["rejected"] = rejected

    if with_llm:
        def answer(qid):
            try:
                return hw4.generate(questions[qid], got[qid])
            except Exception as exc:
                print(f"  generator failed on {qid}: {exc!r}")
                return ""

        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = dict(zip(questions, pool.map(answer, questions)))
        results = [
            {"variant": variant.name, "id": qid, "answer": answer,
             "answerable": qid in answerable,
             "correct": hw4.answer_correct(answer, ground_truth[qid])}
            for qid, answer in answers.items()
        ]
        details.extend(results)
        accuracy = sum(result["correct"] for result in results) / len(questions)
        row["4"] = hw4.POINTS["4"] * accuracy
        row["accuracy"] = accuracy
    else:
        row["4"] = None
        row["accuracy"] = None
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--output", default="ablation.csv")
    parser.add_argument("--with-llm", action="store_true",
                        help="also fill the Task 4 column (needs DEEPSEEK_API_KEY)")
    parser.add_argument("--markdown", action="store_true",
                        help="print the table as markdown for README.md")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    ground_truth = hw4.load_json(folder / "ground_truth.json")
    tables = hw4.load_json(folder / "tables.json") or []
    docs_meta = hw4.load_json(folder / "docs.json") or {}

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    with_llm = args.with_llm and key and key != "your_key_here"
    if args.with_llm and not with_llm:
        print("no usable DEEPSEEK_API_KEY: the Task 4 column stays empty")

    rows = []
    details = []
    for variant in VARIANTS:
        row = score_variant(variant, corpus, questions, ground_truth, tables, docs_meta, with_llm, details)
        rows.append(row)
        print(f"{row['variant']}: 1.1={row['1.1']:.2f} 1.2={row['1.2']:.2f} "
              f"2.1={row['2.1']:.2f} 2.2={row['2.2']:.2f} 2.3={row['2.3']:.2f} "
              f"3={row['3']:.2f} recall={row['recall']:.1%} "
              f"top1={row['top1']:.1%} both={row['both']:.1%}", flush=True)

    fields = ["variant", "change", "1.1", "1.2", "2.1", "2.2", "2.3", "3", "4",
              "recall", "top1", "both", "kept", "rejected", "accuracy"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}")
    if details:
        detail_path = Path(args.output).with_name(Path(args.output).stem + "_answers.csv")
        with detail_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)

    if args.markdown:
        def cell(value, spec):
            return "n/a" if value is None else format(value, spec)

        print()
        print("| Variant | Change | 1.1 | 1.2 | 2.1 | 2.2 | 2.3 | 3 | 4 | Total 1-4 | recall@5 | top-1 | both sides |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            total = sum(row[k] for k in ("1.1", "1.2", "2.1", "2.2", "2.3", "3", "4")
                        if row[k] is not None)
            print(f"| {row['variant']} | {row['change']} | {cell(row['1.1'], '.2f')} | "
                  f"{cell(row['1.2'], '.2f')} | {cell(row['2.1'], '.2f')} | {cell(row['2.2'], '.2f')} | "
                  f"{cell(row['2.3'], '.2f')} | {cell(row['3'], '.2f')} | {cell(row['4'], '.2f')} | "
                  f"{total:.2f} | {row['recall']:.1%} | {row['top1']:.1%} | {row['both']:.1%} |")


if __name__ == "__main__":
    main()
