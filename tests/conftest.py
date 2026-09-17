"""시험판 공통 설정.

시험은 LLM 을 부르지 않는다. 부르면 돈이 들고, 무엇보다 답이 매번 달라 시험이
시험 노릇을 못 한다. LLM 이 없을 때의 길(규칙 편성)이 늘 서 있어야 한다는 것도
여기서 같이 지킨다.

검색은 data/index 와 임베딩 서버가 있어야 돈다. 없으면 그 시험만 건너뛴다 —
없다고 빨간 불을 켜지 않는다.
"""

from __future__ import annotations

import os

import pytest

os.environ["COACH_LLM"] = "0"

from family_fitness_ai.common.settings import settings  # noqa: E402


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings(), "coach_llm", False, raising=False)


def _has_release() -> bool:
    release = settings().release_dir
    return all(
        (release / name).exists()
        for name in ("value_quantiles.csv", "grade_thresholds.csv", "video_clips.csv")
    )


def _has_index() -> bool:
    return (settings().index_dir / "corpus.faiss").exists()


def _has_embedder() -> bool:
    import httpx

    try:
        return httpx.get(settings().embedding_url + "/health", timeout=1.0).is_success
    except Exception:  # noqa: BLE001
        return False


needs_release = pytest.mark.skipif(not _has_release(), reason="data/release 표가 없다")
needs_index = pytest.mark.skipif(not _has_index(), reason="data/index 가 없다")
needs_embedder = pytest.mark.skipif(not _has_embedder(), reason="임베딩 서버가 꺼져 있다")
