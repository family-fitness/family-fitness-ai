"""색인에서 청크를 찾는다 (docs/dev/AI-8 §4 · docs/04 §3).

| | |
|---|---|
| 1차 top-k | 10 |
| 최종 컨텍스트 | 최대 5 |
| 연령 필터 | **언제나** — 끄는 인자를 만들지 않는다 |
| 임계값 | 미달은 컨텍스트에 넣지 않는다 |

**연령 필터는 검색 자체에 걸린다.** FAISS 는 후보를 고르는 단계에서 걸러, 필터를 거치지
않는 경로가 존재하지 않게 한다 (docs/04 §3). `filtered_out` 은 그 때문에 빠진 수를
세어 돌려주는 값이고 — 걸러낸 뒤에 세는 것이지, 세고 나서 거르는 것이 아니다.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from ..common.settings import INDEX_DIR
from . import chunks as K
from .index import INDEX_FILE, MANIFEST_FILE, META_FILE

TOP_K = 10
MAX_CONTEXT = 5


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    source: str
    text: str
    citation_label: str
    citation_url: str
    age_group: str
    score: float


@dataclass(frozen=True)
class Result:
    """찾은 청크와, 빠진 수. 둘이 함께 나가야 0건이 왜 0건인지 알 수 있다."""

    hits: tuple[Hit, ...]
    filtered_out: dict[str, int]


class Corpus:
    """색인과 메타데이터. 행 순서가 벡터 순서다."""

    def __init__(self, index: Any, meta: pd.DataFrame, manifest: dict[str, Any]) -> None:
        self.index = index
        self.meta = meta
        self.manifest = manifest

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> Corpus:
        manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        return cls(
            faiss.read_index(str(index_dir / INDEX_FILE)),
            K.read_csv(index_dir / META_FILE),
            manifest,
        )

    def allowed(self, age_groups: Sequence[str]) -> np.ndarray:
        """그 연령대의 청크와 연령 무관 청크 (docs/04 §2.2 — `age_group = ? OR IS NULL`)."""
        ages = self.meta["age_group"]
        return np.flatnonzero((ages == "") | ages.isin(list(age_groups))).astype("int64")

    def search(
        self,
        vector: np.ndarray,
        age_groups: Sequence[str],
        threshold: float,
        top_k: int = TOP_K,
        limit: int = MAX_CONTEXT,
    ) -> Result:
        """정규화된 질의 벡터 하나로 찾는다. 점수는 코사인 유사도다."""
        allowed = self.allowed(age_groups)
        query = np.asarray(vector, dtype="float32").reshape(1, -1)
        if allowed.size == 0:
            return Result((), {"age_group": int(len(self.meta)), "below_threshold": 0})

        selector = faiss.IDSelectorArray(allowed.size, faiss.swig_ptr(allowed))
        params = faiss.SearchParameters()
        params.sel = selector  # 후보를 고르는 단계에서 거른다 — 뒤에서 걸러내지 않는다
        scores, rows = self.index.search(query, min(top_k, allowed.size), params=params)
        found = [(float(s), int(r)) for s, r in zip(scores[0], rows[0], strict=True) if r >= 0]
        kept = [(s, r) for s, r in found if s >= threshold]
        hits = tuple(self._hit(row, score) for score, row in kept[:limit])
        return Result(
            hits,
            {
                # 연령으로 빠진 청크 수 — 왜 결과가 적은지를 재는 값이다 (docs/03 §4.2)
                "age_group": int(len(self.meta) - allowed.size),
                "below_threshold": len(found) - len(kept),
            },
        )

    def _hit(self, row: int, score: float) -> Hit:
        chunk = self.meta.iloc[row]
        return Hit(
            chunk_id=str(chunk["chunk_id"]),
            source=str(chunk["source"]),
            text=str(chunk["text"]),
            citation_label=str(chunk["citation_label"]),
            citation_url=str(chunk["citation_url"]),
            age_group=str(chunk["age_group"]),
            score=score,
        )
