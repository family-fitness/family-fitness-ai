"""서비스 골격 (docs/01)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from family_fitness_ai.api import readiness
from family_fitness_ai.api.app import app
from family_fitness_ai.common import logging as reqlog
from family_fitness_ai.common.errors import STATUS_OF, ApiError, ErrorCode
from family_fitness_ai.common.settings import Settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_healthz_는_의존을_확인하지_않는다(client: TestClient) -> None:
    # 산출물도 인덱스도 없는 상태에서도 200 이어야 한다 (docs/01 §2.3).
    assert client.get("/healthz").status_code == 200


CHECK_NAMES = {"release", "criteria", "mission_data", "chunks", "sim_threshold", "vector"}


def test_readyz_는_치명_검사만으로_준비를_가른다(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code in (200, 503)
    body = response.json()
    assert set(body["checks"]) == CHECK_NAMES
    fatal = {k: v for k, v in body["checks"].items() if k in readiness.FATAL_CHECKS}
    assert body["ready"] is all(fatal.values())


def test_readyz_는_실패한_이유를_항목별로_돌려준다(tmp_path: Path) -> None:
    """503 하나로 뭉개면 배포 중에 무엇이 안 됐는지 알 수 없다."""
    result = readiness.check(
        Settings(sim_threshold=None, vector_backend="faiss"),
        release_dir=tmp_path / "없다",
        index_dir=tmp_path / "인덱스없다",
    )
    assert result.ready is False
    assert result.checks == dict.fromkeys(CHECK_NAMES, False)
    assert set(result.reasons) == set(result.checks)
    # 못 쓰는 기능을 이름으로 밝힌다
    assert result.degraded == ["chat", "missions"]


def _ready_dir(tmp_path: Path) -> tuple[Path, Path]:
    for name in (
        *readiness.RELEASE_FILES,
        readiness.CRITERIA_FILE,
        readiness.CHUNKS_FILE,
        *readiness.MISSION_FILES,
    ):
        (tmp_path / name).write_text("", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / "faiss.bin").write_text("", encoding="utf-8")
    return tmp_path, index


def _check(release: Path, index: Path, threshold: float | None) -> readiness.Readiness:
    return readiness.check(
        Settings(sim_threshold=threshold, vector_backend="faiss"),
        release_dir=release,
        index_dir=index,
    )


def test_sim_threshold_가_비어_있어도_준비된_것이다(tmp_path: Path) -> None:
    """**임계값은 부수다.** 없으면 `coach/messages` 만 막히고 미션 편성은 돈다.

    전부 통과해야 준비됨이던 옛 규칙은 색인이 없는 새 클론에서 `coach/runs` 까지
    헬스체크로 막았다 — 미션 편성은 임계값도 색인도 임베딩 서버도 쓰지 않는다.
    """
    release, index = _ready_dir(tmp_path)
    empty = _check(release, index, None)
    assert empty.ready is True
    assert empty.checks["sim_threshold"] is False
    assert empty.degraded == ["chat"]

    filled = _check(release, index, 0.35)
    assert filled.ready is True
    assert filled.degraded == []


def test_색인이_없어도_준비된_것이다(tmp_path: Path) -> None:
    """`.gitignore` 가 `*.faiss` 를 막으므로 새 클론에는 색인이 없다."""
    release, _ = _ready_dir(tmp_path)
    result = _check(release, tmp_path / "색인없다", 0.35)
    assert result.ready is True
    assert result.checks["vector"] is False
    assert result.degraded == ["chat"]


def test_미션_재료가_없으면_준비되지_않은_것이다(tmp_path: Path) -> None:
    """**이것은 치명이다** — 없으면 `coach/runs` 가 미션을 하나도 내지 못한다."""
    release, index = _ready_dir(tmp_path)
    for name in readiness.MISSION_FILES:
        (release / name).unlink()
    result = _check(release, index, 0.35)
    assert result.ready is False
    assert result.checks["mission_data"] is False
    assert "missions" in result.degraded


def test_오류는_한_형태다() -> None:
    """docs/03 §2.2 — 성공은 봉투를 씌우지 않고, 실패는 하나의 모양이다."""
    error = ApiError(ErrorCode.ITEM_NOT_ALLOWED, "005 는 받지 않는다")
    assert error.status_code == 400
    assert error.body() == {"error": {"code": "ITEM_NOT_ALLOWED", "message": "005 는 받지 않는다"}}


def test_모든_오류_코드에_상태가_있다() -> None:
    assert set(STATUS_OF) == set(ErrorCode)


def test_ApiError_가_봉투로_변환된다() -> None:
    probe = FastAPI()
    probe.add_exception_handler(ApiError, app.exception_handlers[ApiError])

    @probe.get("/터진다")
    async def boom() -> None:
        raise ApiError(ErrorCode.RUN_IN_PROGRESS, "실행 중인 코치 실행이 있습니다")

    response = TestClient(probe, raise_server_exceptions=False).get("/터진다")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "RUN_IN_PROGRESS"


def test_요청_한_건이_로그_한_줄이_된다(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="family_fitness_ai.request"):
        client.get("/healthz")
    lines = [json.loads(r.message) for r in caplog.records]
    assert len(lines) == 1
    assert lines[0]["path"] == "/healthz"
    assert lines[0]["status"] == 200
    assert "latency_ms" in lines[0]


def test_로그에_넣지_않기로_한_값은_얹히지_않는다() -> None:
    """측정값 원본과 질문 원문은 로그에 넣지 않는다 (docs/01 §5)."""
    reqlog.reset_fields()
    reqlog.add_fields(profile_ref="p_c7a91f")
    assert reqlog.current_fields() == {"profile_ref": "p_c7a91f"}
    with pytest.raises(ValueError, match="로그에 넣지 않기로 한 값"):
        reqlog.add_fields(question="무릎이 아파요")
    with pytest.raises(ValueError, match="로그에 넣지 않기로 한 값"):
        reqlog.add_fields(measurements={"028": 41.3})


def test_설정은_값이_아니라_설정_여부만_내놓는다() -> None:
    settings = Settings(anthropic_api_key="비밀", sim_threshold=None)
    reported = settings.configured()
    assert reported["anthropic_api_key"] is True
    assert reported["sim_threshold"] is False
    assert "비밀" not in json.dumps(reported, ensure_ascii=False)


def test_벡터_백엔드_기본값은_faiss_다() -> None:
    """DB가 서기 전에도 개발이 멈추지 않게 한다 (docs/01 §1). 로컬 `.env` 는 읽지 않는다."""
    assert Settings(_env_file=None).vector_backend == "faiss"


def test_빈_환경변수는_설정하지_않은_것이다(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env.example` 을 복사하면 `SIM_THRESHOLD=` 같은 빈 줄이 남는다. 기동이 막히면 안 된다."""
    monkeypatch.setenv("SIM_THRESHOLD", "")
    monkeypatch.setenv("VECTOR_BACKEND", "")
    settings = Settings(_env_file=None)
    assert settings.sim_threshold is None
    assert settings.vector_backend == "faiss"
