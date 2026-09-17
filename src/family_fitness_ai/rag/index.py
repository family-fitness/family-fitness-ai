"""코퍼스 인덱스. data/index 를 한 번 읽어 들고 있는다.

인덱스를 만드는 일은 서비스 밖이다 — 여기서는 읽기만 한다.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np

from family_fitness_ai.common.settings import settings

#: 코퍼스는 인증을 못 받은 칸을 「미달」로 적어 두었다. 원자료의 이름은 「참가」고,
#: 그 말이 화면에 나가는 말이다 — 읽는 쪽으로 옮겨 놓는다. chunk_id 는 식별자라
#: 그대로 둔다.
_REWRITE = (("미달", "참가"),)


def _readable(text: str) -> str:
    for before, after in _REWRITE:
        text = text.replace(before, after)
    return text


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str  # criteria · prescription · video
    text: str
    citation_label: str
    citation_url: str
    age_group: str
    factors: tuple[str, ...]
    grade: str

    def citation(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "label": self.citation_label,
            "chunk_id": self.chunk_id,
            "url": self.citation_url or None,
        }


@dataclass(frozen=True)
class Corpus:
    index: faiss.Index
    chunks: tuple[Chunk, ...]
    meta: dict[str, object]

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        query = np.asarray([vector], dtype="float32")
        scores, ids = self.index.search(query, min(k, len(self.chunks)))
        out = []
        for chunk_id, score in zip(ids[0], scores[0], strict=True):
            if chunk_id < 0:
                continue
            out.append((self.chunks[int(chunk_id)], float(score)))
        return out

    def by_id(self, chunk_id: str) -> Chunk | None:
        return self._lookup.get(chunk_id)

    @property
    def _lookup(self) -> dict[str, Chunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}


@lru_cache
def corpus() -> Corpus:
    directory = settings().index_dir
    index = faiss.read_index(str(directory / "corpus.faiss"))
    meta = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))

    csv.field_size_limit(sys.maxsize)
    chunks = []
    with (directory / "corpus_meta.csv").open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            factors = tuple(f for f in (row.get("fitness_factors") or "").split("·") if f)
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    text=_readable(row["text"]),
                    citation_label=_readable(row["citation_label"]),
                    citation_url=row.get("citation_url") or "",
                    age_group=row.get("age_group") or "",
                    factors=factors,
                    grade=_readable(row.get("grade") or ""),
                )
            )

    if index.ntotal != len(chunks):
        raise RuntimeError(f"인덱스와 메타의 줄 수가 다르다: {index.ntotal} vs {len(chunks)}")
    return Corpus(index=index, chunks=tuple(chunks), meta=meta)
