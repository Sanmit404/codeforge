"""BM25 keyword scoring and rank fusion.

Dense embeddings are weak on exact identifiers like `refresh_token`, so the index
runs a keyword pass as well and merges the two rankings.
"""

from __future__ import annotations

import math
import re
from collections import Counter

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
WORD = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word pieces, splitting camelCase and snake_case identifiers."""
    return [word.lower() for word in WORD.findall(CAMEL_BOUNDARY.sub(" ", text))]


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        tokenized = [tokenize(document) for document in documents]
        self.lengths = [len(tokens) for tokens in tokenized]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if tokenized else 1.0
        self.frequencies = [Counter(tokens) for tokens in tokenized]

        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        total = len(tokenized)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        """Score every document against the query."""
        terms = tokenize(query)
        scale = self.average_length or 1.0
        results = []
        for frequency, length in zip(self.frequencies, self.lengths):
            total = 0.0
            for term in terms:
                count = frequency.get(term, 0)
                if not count:
                    continue
                denominator = count + self.k1 * (1 - self.b + self.b * length / scale)
                total += self.idf.get(term, 0.0) * count * (self.k1 + 1) / denominator
            results.append(total)
        return results


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge ranked id lists by position, so the two score scales never have to match."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, key in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + position + 1)
    return sorted(fused, key=lambda key: fused[key], reverse=True)
