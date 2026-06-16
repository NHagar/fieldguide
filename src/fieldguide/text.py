"""Text normalization, tokenization, and small lexical helpers."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable


STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{1,}|\$?\d[\d,]*(?:\.\d+)?")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")


def stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()[:length].upper()
    return f"{prefix}-{digest}"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def estimate_tokens(text_or_obj: object) -> int:
    if not isinstance(text_or_obj, str):
        text_or_obj = str(text_or_obj)
    return max(1, math.ceil(len(text_or_obj) / 4))


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    terms: list[str] = []
    for match in WORD_RE.finditer(text.lower()):
        term = match.group(0).strip("'_-")
        if len(term) < 2:
            continue
        if not keep_stopwords and term in STOPWORDS:
            continue
        if len(term) > 40:
            continue
        terms.append(term)
    return terms


def lexical_terms(text: str) -> list[str]:
    words = tokenize(text)
    terms = list(words)
    for left, right in zip(words, words[1:]):
        if left not in STOPWORDS and right not in STOPWORDS:
            terms.append(f"{left} {right}")
    return terms


def top_terms(texts: Iterable[str], limit: int = 8, *, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(lexical_terms(text))
    for term in list(counts):
        if term in exclude or len(term) < 3:
            del counts[term]
    return [term for term, _ in counts.most_common(limit)]


def term_counts(text: str) -> Counter[str]:
    return Counter(lexical_terms(text))


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def normalized_vector(counts: Counter[str], idf: dict[str, float], *, max_terms: int = 400) -> dict[str, float]:
    weighted: dict[str, float] = {}
    total = sum(counts.values()) or 1
    for term, count in counts.most_common(max_terms):
        weighted[term] = (count / total) * idf.get(term, 1.0)
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm:
        weighted = {term: value / norm for term, value in weighted.items()}
    return weighted


def average_vector(vectors: Iterable[dict[str, float]]) -> dict[str, float]:
    total: Counter[str] = Counter()
    count = 0
    for vector in vectors:
        count += 1
        total.update(vector)
    if not count:
        return {}
    averaged = {term: value / count for term, value in total.items()}
    norm = math.sqrt(sum(value * value for value in averaged.values()))
    if norm:
        averaged = {term: value / norm for term, value in averaged.items()}
    return averaged


def trim_to_range(text: str, start: int, end: int) -> tuple[int, int, str]:
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end, text[start:end]


def make_excerpt_range(text: str, hit_start: int, hit_end: int, target_chars: int, max_chars: int) -> tuple[int, int, str]:
    target_chars = min(target_chars, max_chars)
    center = (hit_start + hit_end) // 2
    start = max(0, center - target_chars // 2)
    end = min(len(text), start + target_chars)
    start = max(0, end - target_chars)

    left_boundary = text.rfind("\n", 0, start + 1)
    if left_boundary >= 0 and start - left_boundary < 120:
        start = left_boundary + 1
    right_boundary = text.find("\n", end)
    if right_boundary >= 0 and right_boundary - end < 120:
        end = right_boundary
    if end - start > max_chars:
        end = start + max_chars
    return trim_to_range(text, start, end)
