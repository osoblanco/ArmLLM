"""The tools the agent is given, and the corpus they read from.

Four tools, on purpose:

    search(query, k)   BM25 over the corpus -> ranked snippets
    read_doc(doc_id)   the full text of one document
    calc(expression)   arithmetic, because models are bad at it and tools aren't
    finish(answer)     the stopping rule, expressed as a tool

`finish` being a tool rather than "the model stopped talking" is the whole
point of Act I. A loop with no `finish` has nothing authorised to say stop.

Retrieval is BM25, not embeddings: deterministic, dependency-free, and
inspectable. It also has an honest failure mode you will meet at 09:05 — a
document that repeats a phrase many times outranks a short notice that
supersedes it. Nothing here knows what "superseded" means. Nothing was asked to.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

K1 = 1.5
B = 0.75


@dataclass
class Doc:
    id: str
    title: str
    published: str
    text: str
    tokens: list[str] = None  # type: ignore[assignment]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def load_corpus(corpus_dir: str | None = None) -> tuple[Doc, ...]:
    directory = Path(corpus_dir) if corpus_dir else CORPUS_DIR
    docs: list[Doc] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        published = ""
        m = re.search(r"^Published:\s*(\S+)", text, re.MULTILINE)
        if m:
            published = m.group(1)
        docs.append(Doc(id=path.stem, title=title, published=published, text=text,
                        tokens=_tokenize(text)))
    return tuple(docs)


class Corpus:
    def __init__(self, corpus_dir: str | None = None) -> None:
        self.docs = list(load_corpus(corpus_dir))
        self.by_id = {d.id: d for d in self.docs}
        self.avgdl = sum(len(d.tokens) for d in self.docs) / max(1, len(self.docs))
        self.df: dict[str, int] = {}
        for d in self.docs:
            for term in set(d.tokens):
                self.df[term] = self.df.get(term, 0) + 1
        self.n = len(self.docs)

    def add_docs(self, directory: str | Path) -> int:
        """Used by milestone 4 to drop poisoned documents into the corpus."""
        added = 0
        for path in sorted(Path(directory).glob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip()
            doc = Doc(id=path.stem, title=title, published="", text=text,
                      tokens=_tokenize(text))
            if doc.id not in self.by_id:
                self.docs.append(doc)
                self.by_id[doc.id] = doc
                for term in set(doc.tokens):
                    self.df[term] = self.df.get(term, 0) + 1
                self.n += 1
                added += 1
        self.avgdl = sum(len(d.tokens) for d in self.docs) / max(1, len(self.docs))
        return added

    def bm25(self, query: str, k: int = 5) -> list[tuple[Doc, float]]:
        q = _tokenize(query)
        scored: list[tuple[Doc, float]] = []
        for d in self.docs:
            dl = len(d.tokens)
            tf: dict[str, int] = {}
            for term in d.tokens:
                tf[term] = tf.get(term, 0) + 1
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                freq = tf[term]
                score += idf * (freq * (K1 + 1)) / (freq + K1 * (1 - B + B * dl / self.avgdl))
            if score > 0:
                scored.append((d, score))
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]


_CORPUS: Corpus | None = None


def corpus() -> Corpus:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = Corpus()
    return _CORPUS


def reset_corpus(with_injections: bool = False) -> Corpus:
    global _CORPUS
    load_corpus.cache_clear()
    _CORPUS = Corpus()
    if with_injections:
        _CORPUS.add_docs(Path(__file__).resolve().parent / "injected")
    return _CORPUS


# --------------------------------------------------------------------------
# The tools themselves
# --------------------------------------------------------------------------


def search(query: str, k: int = 5) -> list[dict]:
    hits = corpus().bm25(query, k=k)
    out = []
    for doc, score in hits:
        body = doc.text.split("Source:", 1)[-1]
        snippet = re.sub(r"\s+", " ", body).strip()[:280]
        out.append({
            "id": doc.id,
            "title": doc.title,
            "published": doc.published,
            "score": round(score, 3),
            "snippet": snippet,
        })
    return out


def read_doc(doc_id: str) -> str:
    doc_id = str(doc_id).strip()
    # A document is allowed to ask for anything. That does not mean we comply.
    # injected/inj-02.md asks the agent to read `../../.env`; this is where that
    # request stops being interesting.
    if not re.fullmatch(r"[A-Za-z0-9._-]+", doc_id):
        return f"ERROR: refused. '{doc_id}' is not a document id in this corpus."
    doc = corpus().by_id.get(doc_id)
    if doc is None:
        return f"ERROR: no document with id '{doc_id}'."
    return doc.text


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}


def calc(expression: str) -> str:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    cleaned = str(expression).replace(",", "").replace(" ", "")
    try:
        value = ev(ast.parse(cleaned, mode="eval"))
    except Exception as exc:  # noqa: BLE001 - the agent should see the error text
        return f"ERROR: {exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def finish(answer: str) -> str:
    return str(answer)


TOOLS = {"search": search, "read_doc": read_doc, "calc": calc, "finish": finish}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Full-text search over the Dvin municipal document corpus. Returns ranked "
                "documents with id, title, publication date and a short snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms."},
                    "k": {"type": "integer", "description": "How many results (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": "Return the full text of one document, by its id.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate an arithmetic expression, e.g. '2000 + 1500'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Give the final answer and stop. If the documents do not contain the answer, "
                "say so explicitly rather than guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]
