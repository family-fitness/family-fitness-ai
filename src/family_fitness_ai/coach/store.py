"""코치 실행 결과를 폴링 사이에 들고 있는다 (docs/03 §5.2·§5.3).

**영속 상태의 정본은 백엔드다** (`coach_runs` 표 · docs/01 §3.1 — 그래프에
체크포인터를 붙이지 않는다). 여기 있는 것은 `POST` 가 낸 결과를 곧 이어질 `GET`
까지 들고 있는 임시 자리다.

> **태스크가 둘이면 폴링이 빈다.** `POST` 를 받은 프로세스와 `GET` 을 받은
> 프로세스가 다르면 404 다. 배포에서 태스크를 하나로 두거나 공용 저장소를
> 붙여야 한다 — `docs/01` 에 제약으로 적는다.

오래된 것과 넘치는 것을 함께 버린다. 시각은 **호출자가 넣는다** — 시험이 시계를
기다리지 않게 한다.
"""

from __future__ import annotations

import secrets
from collections import OrderedDict
from dataclasses import dataclass

from ..api.coach_schemas import MAX_RUN_ID, CoachRunResult

RUN_PREFIX = "cr_"
# `cr_` + 16자 = 19자. `coach_runs.ai_run_id` 가 varchar(40) 이다
RUN_RANDOM_CHARS = 16
TTL_SEC = 600.0
MAX_RUNS = 256


def new_run_id() -> str:
    """`cr_` 접두 (docs/03 §5.2). 길이가 계약 상한 안임을 단정으로 잠근다."""
    run_id = RUN_PREFIX + secrets.token_hex(RUN_RANDOM_CHARS // 2)
    assert len(run_id) <= MAX_RUN_ID
    return run_id


@dataclass
class _Entry:
    result: CoachRunResult
    stored_at: float


class RunStore:
    """가장 오래된 것부터 버린다. 스레드 하나를 전제한다 (uvicorn 워커 안)."""

    def __init__(self, ttl_sec: float = TTL_SEC, max_runs: int = MAX_RUNS) -> None:
        self._runs: OrderedDict[str, _Entry] = OrderedDict()
        self.ttl_sec = ttl_sec
        self.max_runs = max_runs

    def put(self, result: CoachRunResult, now: float) -> None:
        self._evict(now)
        self._runs[result.run_id] = _Entry(result, now)
        self._runs.move_to_end(result.run_id)
        while len(self._runs) > self.max_runs:
            self._runs.popitem(last=False)

    def get(self, run_id: str, now: float) -> CoachRunResult | None:
        self._evict(now)
        entry = self._runs.get(run_id)
        return None if entry is None else entry.result

    def _evict(self, now: float) -> None:
        stale = [k for k, v in self._runs.items() if now - v.stored_at > self.ttl_sec]
        for key in stale:
            del self._runs[key]

    def __len__(self) -> int:
        return len(self._runs)
