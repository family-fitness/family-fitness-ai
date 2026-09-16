"""색인 전 검사와 연령 필터 검색 (docs/dev/AI-8 §3·§4 · docs/04 §3).

임베딩 서버 없이 돈다 — 벡터를 직접 만든다. 모델이 무엇을 가깝다고 보는지는 여기서
재지 않는다. 그것은 `SIM_THRESHOLD` 측정이 잰다 (docs/dev/AI-8 §5).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from family_fitness_ai.rag import index as I
from family_fitness_ai.rag import search as S

DIM = 4


def chunks(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """(chunk_id, source, age_group) — 본문은 검사에 걸리지 않을 만큼만 채운다."""
    return pd.DataFrame(
        [
            {
                "chunk_id": chunk_id,
                "source": source,
                "text": f"{chunk_id} 본문",
                "citation_label": f"인용 {chunk_id}",
                "citation_url": "",
                "age_group": age_group,
                "fitness_factors": "",
                "grade": "",
            }
            for chunk_id, source, age_group in rows
        ]
    )


class FakeEmbedder:
    """청크마다 한 축만 1인 벡터. 무엇이 가까운지를 시험이 직접 정한다."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def version(self) -> str:
        return "embed:fake/4"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for i in range(len(texts)):
            vectors[i, i % self.dim] = 1.0
        return vectors

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        return [100] * len(texts)


def test_겹치는_chunk_id_는_색인하지_않는다() -> None:
    """재색인 때 같은 원문이 같은 id 여야 저장된 인용이 끊기지 않는다 (docs/04 §1)."""
    frame = chunks(("video:v1", "video", "유아기"), ("video:v1", "video", "유아기"))
    problems = I.check(frame, [100, 100])
    assert problems and "chunk_id 가 겹친다" in problems[0]


def test_연령이_빈_처방_영상_청크는_색인하지_않는다() -> None:
    """이 검사가 `age_group IS NULL` 을 연령 무관으로 쓰는 것을 안전하게 만든다 (docs/04 §2.2)."""
    frame = chunks(("prescription:p", "prescription", ""), ("criteria:c", "criteria", ""))
    problems = I.check(frame, [100, 100])
    # 기준표는 연령 무관이라 비어 있어도 된다
    assert len(problems) == 1
    assert "prescription:p" in problems[0]


def test_너무_짧거나_긴_청크를_짚는다() -> None:
    frame = chunks(("a:1", "criteria", ""), ("b:2", "criteria", ""))
    problems = I.check(frame, [I.MIN_TOKENS - 1, I.MAX_TOKENS + 1])
    assert len(problems) == 2
    assert "50토큰 미만" in problems[0] and "400토큰 초과" in problems[1]


def test_검사를_통과하지_못하면_색인하지_않는다() -> None:
    frame = chunks(("video:v1", "video", ""))
    with pytest.raises(I.IndexRefused):
        I.build(frame, FakeEmbedder(), "2026-09-16.1")


def test_색인은_판_정보를_함께_남긴다() -> None:
    frame = chunks(("video:v1", "video", "유아기"), ("criteria:c", "criteria", ""))
    index, manifest = I.build(frame, FakeEmbedder(), "2026-09-16.1")
    assert index.ntotal == 2
    assert manifest["corpus_version"] == "2026-09-16.1"
    assert manifest["embedding_version"] == "embed:fake/4"
    assert manifest["sources"] == {"criteria": 1, "video": 1}


def corpus() -> S.Corpus:
    frame = chunks(
        ("video:kid", "video", "유아기"),
        ("prescription:adult", "prescription", "성인"),
        ("criteria:any", "criteria", ""),
    )
    index, manifest = I.build(frame, FakeEmbedder(), "2026-09-16.1")
    return S.Corpus(index, frame, manifest)


def test_다른_연령대_청크는_검색에서_빠진다() -> None:
    """연령 필터는 정확성이 아니라 안전 문제다 (docs/04 §3)."""
    kid = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")  # prescription:adult 의 벡터
    result = corpus().search(kid, ["유아기"], threshold=0.5)
    assert [h.chunk_id for h in result.hits] == []
    assert result.filtered_out["age_group"] == 1  # 성인 청크가 후보에서 빠졌다


def test_연령_무관_청크는_어느_연령대에서도_살아남는다() -> None:
    any_chunk = np.array([0.0, 0.0, 1.0, 0.0], dtype="float32")
    result = corpus().search(any_chunk, ["유아기"], threshold=0.5)
    assert [h.chunk_id for h in result.hits] == ["criteria:any"]
    assert result.hits[0].score == pytest.approx(1.0)


def test_임계값_미달은_컨텍스트에_넣지_않고_센다() -> None:
    """0건인데 LLM 을 호출하는 경로를 만들지 않는다 (docs/04 §3)."""
    between = np.array([0.6, 0.0, 0.8, 0.0], dtype="float32")  # 무관 청크와 0.8, 영상과 0.6
    result = corpus().search(between, ["유아기"], threshold=0.9)
    assert result.hits == ()
    assert result.filtered_out["below_threshold"] == 2


def test_컨텍스트는_최대_다섯_청크다() -> None:
    frame = chunks(*[(f"criteria:{i}", "criteria", "") for i in range(8)])
    index, manifest = I.build(frame, FakeEmbedder(dim=8), "2026-09-16.1")
    result = S.Corpus(index, frame, manifest).search(
        np.ones(8, dtype="float32") / np.sqrt(8), ["유아기"], threshold=0.0
    )
    assert len(result.hits) == S.MAX_CONTEXT
