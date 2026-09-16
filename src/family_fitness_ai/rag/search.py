"""색인에서 청크를 찾는다 (docs/04 · docs/04 §3).

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
from ..common.types import AGE_GROUPS
from . import chunks as K
from .index import INDEX_FILE, MANIFEST_FILE, META_FILE

TOP_K = 10
MAX_CONTEXT = 5


class CorpusMismatch(RuntimeError):
    """색인·메타·판 정보가 서로 맞지 않는다. 다시 색인해야 한다."""


def _age_list(age_groups: Sequence[str]) -> list[str]:
    """연령대 목록을 받는다. **문자열 하나를 받으면 막는다.**

    `"유아기"` 를 넘기면 파이썬은 글자 단위로 훑어 아무것도 맞지 않는다 — 오류 없이
    결과만 비어, 연령 필터가 과하게 걸린 것을 눈치채기 어렵다.
    """
    if isinstance(age_groups, str):
        raise ValueError(f"연령대는 목록이어야 한다 — {age_groups!r} 하나가 왔다")
    wanted = [str(a) for a in age_groups]
    unknown = [a for a in wanted if a not in AGE_GROUPS]
    if unknown:
        raise ValueError(f"모르는 연령대 {unknown} — {list(AGE_GROUPS)} 중에서 쓴다")
    return wanted


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
        index = faiss.read_index(str(index_dir / INDEX_FILE))
        meta = K.read_csv(index_dir / META_FILE)
        # 셋이 어긋나면 검색은 조용히 엉뚱한 청크를 짚는다 — 행 번호가 벡터 번호이기
        # 때문이다. 읽을 때 한 번 맞춰 보는 값이 싸다.
        if index.ntotal != len(meta):
            raise CorpusMismatch(f"색인 {index.ntotal}개 · 메타 {len(meta)}줄")
        if index.d != int(manifest.get("dim", index.d)):
            raise CorpusMismatch(f"색인 {index.d}차원 · 판 정보 {manifest['dim']}차원")
        return cls(index, meta, manifest)

    def expects(self, embedder_version: str) -> None:
        """색인을 만든 임베딩 판과 지금 쓰는 판이 같은지 본다 (docs/04 §2.2).

        모델을 바꾸면 전량 재색인이다. 그냥 돌면 점수가 뜻을 잃는다.
        """
        built = str(self.manifest.get("embedding_version", ""))
        if built and built != embedder_version:
            raise CorpusMismatch(f"색인은 {built} 로 만들었다 — 지금은 {embedder_version}")

    def allowed(self, age_groups: Sequence[str]) -> np.ndarray:
        """그 연령대의 청크와 연령 무관 청크 (docs/04 §2.2 — `age_group = ? OR IS NULL`)."""
        wanted = _age_list(age_groups)
        ages = self.meta["age_group"]
        return np.flatnonzero((ages == "") | ages.isin(wanted)).astype("int64")

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
            return Result(
                (), {"age_group": int(self._aged_out(query, top_k, None)), "below_threshold": 0}
            )

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
                "age_group": int(self._aged_out(query, top_k, allowed)),
                "below_threshold": len(found) - len(kept),
            },
        )

    def _aged_out(self, query: np.ndarray, top_k: int, allowed: np.ndarray | None) -> int:
        """연령 필터가 없었다면 상위에 들었을 청크 중 몇 개가 빠졌나 (docs/03 §4.2).

        **코퍼스 전체에서 연령이 다른 청크 수를 세지 않는다.** 그 값은 질의와 무관한
        상수라 "왜 결과가 적은가"를 재지 못하고, 같은 표의 `below_threshold` 와 분모가
        달라 나란히 읽을 수 없다. 같은 top-k 창에서 잰다.
        """
        _, rows = self.index.search(query, min(top_k, self.index.ntotal))
        candidates = {int(r) for r in rows[0] if r >= 0}
        if allowed is None:
            return len(candidates)
        return len(candidates - {int(r) for r in allowed})

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
