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


# **치명과 부수를 가른다.** 하나라도 false 면 503 이던 규칙은 `coach/runs` 까지 막았다 —
# 미션 편성은 커밋된 CSV 조회뿐이라 색인도 임계값도 임베딩 서버도 쓰지 않는다.
# `.gitignore` 가 `*.faiss` 를 막으므로 새 클론·컨테이너에는 색인이 아예 없고,
# 그 상태로 전체를 막으면 도는 기능까지 헬스체크로 죽는다 (2026-09-16 실측).
#
# 치명 = 이것이 없으면 **어떤 엔드포인트도** 답하지 못한다.
FATAL_CHECKS = frozenset({"release", "criteria", "mission_data"})

# 부수 검사가 무엇을 못 쓰게 하나. `/readyz` 가 `degraded` 로 이름을 밝힌다.
DEGRADES: dict[str, str] = {
    "sim_threshold": "chat",
    "vector": "chat",
    "chunks": "chat",
    "mission_data": "missions",
}

# 편성이 읽는 것. `missions.csv` 가 추천의 검색 대상이고 `prescription_cells.csv`
# 가 인용 근거다 — 하나라도 없으면 `coach/runs` 가 미션을 못 낸다.
MISSION_FILES = ("prescription_cells.csv", "missions.csv")

# 색인의 재료. 이것이 있으면 새 클론에서 색인을 다시 구울 수 있다 (docs/04 §5).
CHUNKS_FILE = "chunks.csv"


@dataclass(frozen=True)
class Readiness:
    checks: dict[str, bool]
    reasons: dict[str, str]

    @property
    def ready(self) -> bool:
        """**치명 검사만 본다.** 부수가 빠진 것은 `degraded` 로 알린다."""
        return all(ok for name, ok in self.checks.items() if name in FATAL_CHECKS)

    @property
    def degraded(self) -> list[str]:
        """못 쓰는 기능 이름. 준비됐어도 비어 있지 않을 수 있다."""
        return sorted(
            {DEGRADES[name] for name, ok in self.checks.items() if not ok and name in DEGRADES}
        )

    def body(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "checks": self.checks,
            "reasons": self.reasons,
            "degraded": self.degraded,
        }


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

    missing_mission = [f for f in MISSION_FILES if not (release_dir / f).exists()]
    record(
        "mission_data",
        not missing_mission,
        f"미션 재료가 없다: {', '.join(missing_mission)}",
    )

    record(
        "chunks",
        (release_dir / CHUNKS_FILE).exists(),
        f"청크가 없다 — 색인을 다시 세울 수 없다: {release_dir / CHUNKS_FILE}",
    )

    # docs/04 §3 — 값이 없으면 임계 없는 검색이 되어 무관한 청크가 근거로 실린다.
    # **부수다** — 미션 편성은 임계값을 쓰지 않는다. 없으면 `coach/messages` 만 막힌다.
    record("sim_threshold", settings.sim_threshold is not None, "SIM_THRESHOLD 가 비어 있다")

    record(*_vector(settings, index_dir))
    return Readiness(checks=checks, reasons=reasons)


def _vector(settings: Settings, index_dir: Path) -> tuple[str, bool, str]:
    """벡터 백엔드 연결. 두 백엔드가 같은 이름의 검사를 낸다 (docs/01 §1).

    pgvector 확인은 배포에서 채운다 ([AI-12](../../../docs/01)) —
    지금은 드라이버를 의존성에 넣지 않았다 (docs/01). AI-8 은 FAISS 경로만
    세웠다. 색인이 서기 전까지 `/readyz` 는 준비되지 않았다고 답하고, 그것이 사실이다.
    """
    if settings.vector_backend == "faiss":
        ok = index_dir.exists() and any(index_dir.iterdir())
        return "vector", ok, f"로컬 인덱스가 없다: {index_dir}"
    return "vector", False, "pgvector 연결 확인은 아직 없다 — 배포(AI-12)에서 채운다"
