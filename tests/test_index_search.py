"""색인 전 검사와 연령 필터 검색 (docs/04 · docs/04 §3).

임베딩 서버 없이 돈다 — 벡터를 직접 만든다. 모델이 무엇을 가깝다고 보는지는 여기서
재지 않는다. 그것은 `SIM_THRESHOLD` 측정이 잰다 (docs/05 §4.2).
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


def test_연령_필터는_후보를_고르는_단계에_있다() -> None:
    """후처리로 거르면 이 시험이 깨진다 (docs/04 §3).

    성인 청크가 상위를 다 차지하도록 만들어 두고 유아기로 찾는다. 뒤에서 걸러내는
    구현이라면 1차 top-k 가 성인으로만 차서 결과가 비지만, 후보 단계에서 걸러내면
    아이 청크가 그 자리를 채운다.
    """
    frame = chunks(
        *[(f"prescription:adult{i}", "prescription", "성인") for i in range(S.TOP_K)],
        ("video:kid", "video", "유아기"),
    )
    index, manifest = I.build(frame, FakeEmbedder(dim=S.TOP_K + 1), "2026-09-16.1")
    kid_vector = np.zeros(S.TOP_K + 1, dtype="float32")
    kid_vector[S.TOP_K % (S.TOP_K + 1)] = 1.0

    result = S.Corpus(index, frame, manifest).search(kid_vector, ["유아기"], threshold=0.5)
    assert [h.chunk_id for h in result.hits] == ["video:kid"]


def test_연령으로_빠진_수는_같은_창에서_센다() -> None:
    """`filtered_out` 두 값의 분모가 같아야 나란히 읽힌다 (docs/03 §4.2).

    코퍼스 전체의 성인 청크 수를 세면 질의와 무관한 상수가 나온다.
    """
    adults = S.TOP_K + 5
    frame = chunks(
        *[(f"prescription:adult{i}", "prescription", "성인") for i in range(adults)],
        ("video:kid", "video", "유아기"),
    )
    dim = adults + 1
    index, manifest = I.build(frame, FakeEmbedder(dim=dim), "2026-09-16.1")
    # 앞 TOP_K 개 성인 청크 쪽으로 기울인 질의 — 연령 필터가 없으면 top-k 를 그들이 채운다
    vector = np.zeros(dim, dtype="float32")
    vector[: S.TOP_K] = 1.0
    vector /= np.linalg.norm(vector)

    result = S.Corpus(index, frame, manifest).search(vector, ["유아기"], threshold=0.0)
    # 상수(성인 청크 15개)가 아니라 top-k 창에서 빠진 수여야 한다
    assert result.filtered_out["age_group"] == S.TOP_K


def test_연령대를_문자열_하나로_넘기면_막는다() -> None:
    """`"유아기"` 는 글자 단위로 훑여 아무것도 맞지 않는다 — 조용히 비면 안 된다."""
    with pytest.raises(ValueError, match="목록"):
        corpus().search(np.zeros(DIM, dtype="float32"), "유아기", threshold=0.5)  # type: ignore[arg-type]


def test_모르는_연령대는_막는다() -> None:
    with pytest.raises(ValueError, match="모르는 연령대"):
        corpus().search(np.zeros(DIM, dtype="float32"), ["중장년"], threshold=0.5)


def test_색인과_메타가_어긋나면_읽지_않는다(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """행 번호가 벡터 번호다 — 어긋난 채로 돌면 엉뚱한 청크를 인용한다."""
    frame = chunks(("video:v1", "video", "유아기"), ("criteria:c", "criteria", ""))
    index, manifest = I.build(frame, FakeEmbedder(), "2026-09-16.1")
    I.save(index, frame, manifest, tmp_path)
    (tmp_path / I.META_FILE).write_text(frame.head(1).to_csv(index=False), encoding="utf-8-sig")
    with pytest.raises(S.CorpusMismatch):
        S.Corpus.load(tmp_path)


def test_다른_임베딩_판으로_만든_색인을_짚는다() -> None:
    """모델을 바꾸면 전량 재색인이다 (docs/04 §2.2). 그냥 돌면 점수가 뜻을 잃는다."""
    with pytest.raises(S.CorpusMismatch, match="embed:fake/4"):
        corpus().expects("embed:bge-m3/1024")
    corpus().expects("embed:fake/4")
