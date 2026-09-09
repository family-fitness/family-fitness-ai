"""준비성 판단 (docs/01 §2.3).

`/readyz` 는 `/healthz` 와 다르다. 프로세스가 살아 있어도 기준표나 벡터 인덱스를
못 읽으면 요청을 받으면 안 된다. 그 상태로 받으면 빈 결과가 나가고, 그것이
"AI가 답을 못 한다"로 기록된다.

**확인한 것을 항목별로 돌려준다.** 준비되지 않은 이유가 503 하나로 뭉개지면
배포 중에 무엇이 안 됐는지 알 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..common.settings import INDEX_DIR, RELEASE_DIR, Settings

# stats/assess.py 의 Reference 가 읽는 것과 같다. 하나라도 없으면 채점이 안 된다.
RELEASE_FILES = (
    "age_band_score_summary.csv",
    "age_band_score_distribution.csv",
    "age_band_value_quantiles.csv",
)
CRITERIA_FILE = "grade_thresholds.csv"


@dataclass(frozen=True)
class Readiness:
    checks: dict[str, bool]
    reasons: dict[str, str]

    @property
    def ready(self) -> bool:
        return all(self.checks.values())

    def body(self) -> dict[str, object]:
        return {"ready": self.ready, "checks": self.checks, "reasons": self.reasons}


def check(
    settings: Settings,
    *,
    release_dir: Path = RELEASE_DIR,
    index_dir: Path = INDEX_DIR,
) -> Readiness:
    checks: dict[str, bool] = {}
    reasons: dict[str, str] = {}

    def record(name: str, ok: bool, reason: str) -> None:
        checks[name] = ok
        if not ok:
            reasons[name] = reason

    missing = [f for f in RELEASE_FILES if not (release_dir / f).exists()]
    record("release", not missing, f"산출물이 없다: {', '.join(missing)}")

    record(
        "criteria",
        (release_dir / CRITERIA_FILE).exists(),
        f"기준표가 없다: {release_dir / CRITERIA_FILE}",
    )

    # docs/04 §6 ② — 값이 없으면 임계 없는 검색이 되어 무관한 청크가 근거로 실린다.
    record("sim_threshold", settings.sim_threshold is not None, "SIM_THRESHOLD 가 비어 있다")

    record(*_vector(settings, index_dir))
    return Readiness(checks=checks, reasons=reasons)


def _vector(settings: Settings, index_dir: Path) -> tuple[str, bool, str]:
    """벡터 백엔드 연결. 두 백엔드가 같은 이름의 검사를 낸다 (docs/01 §1).

    pgvector 확인은 AI-8 이 채운다 — 지금은 드라이버를 의존성에 넣지 않았다
    (docs/dev/AI-3 §4). 색인이 서기 전까지 `/readyz` 는 준비되지 않았다고 답하고,
    그것이 사실이다.
    """
    if settings.vector_backend == "faiss":
        ok = index_dir.exists() and any(index_dir.iterdir())
        return "vector", ok, f"로컬 인덱스가 없다: {index_dir}"
    return "vector", False, "pgvector 연결 확인은 아직 없다 — AI-8 이 채운다"
