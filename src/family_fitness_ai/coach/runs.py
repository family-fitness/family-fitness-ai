"""코치 실행 보관.

편성은 15~40초가 걸려 비동기다. POST 가 run_id 를 내주고, 실제 일은 뒤에서 돈다.
호출자는 1.5초 간격으로 GET 해서 끝나기를 기다린다.

메모리에만 둔다 — **AI 는 DB 에 쓰지 않는다.** 서비스가 내려가면 실행도 사라지고,
그건 호출자가 다시 부르면 되는 일이다. 저장과 승인은 전부 호출자 몫이다.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from family_fitness_ai.coach import verify
from family_fitness_ai.coach.compose import Constraints, Plan, RunProfile, Step, build
from family_fitness_ai.common.errors import run_in_progress, run_not_found

#: 이만큼 지난 실행은 버린다. 폴링이 끝난 뒤에도 잠깐은 남겨 둔다.
TTL_SEC = 30 * 60


def new_run_id() -> str:
    stamp = int(time.time() * 1000).to_bytes(6, "big")
    tail = os.urandom(5)
    return "cr_" + base64.b32encode(stamp + tail).decode().rstrip("=")


@dataclass
class Run:
    run_id: str
    profile_refs: tuple[str, ...]
    status: str = "running"
    steps: list[Step] = field(default_factory=list)
    proposal: dict[str, Any] | None = None
    refused: bool = False
    refusal_reason: str | None = None
    created_at: float = field(default_factory=time.time)

    def dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "steps": [step.dict() for step in self.steps],
            "proposal": self.proposal,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
        }


class Store:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def _sweep(self) -> None:
        stale = time.time() - TTL_SEC
        for run_id, run in list(self._runs.items()):
            if run.created_at < stale:
                del self._runs[run_id]

    def start(
        self,
        profiles: list[RunProfile],
        start_date: date,
        weeks: int,
        constraints: Constraints,
    ) -> Run:
        refs = tuple(profile.ref for profile in profiles)
        with self._lock:
            self._sweep()
            busy = any(
                run.status == "running" and set(run.profile_refs) & set(refs)
                for run in self._runs.values()
            )
            if busy:
                raise run_in_progress()
            run = Run(run_id=new_run_id(), profile_refs=refs)
            self._runs[run.run_id] = run

        worker = threading.Thread(
            target=self._work,
            args=(run, profiles, start_date, weeks, constraints),
            daemon=True,
        )
        worker.start()
        return run

    def _work(
        self,
        run: Run,
        profiles: list[RunProfile],
        start_date: date,
        weeks: int,
        constraints: Constraints,
    ) -> None:
        try:
            plan = build(profiles, start_date, weeks, constraints)
        except Exception as error:  # noqa: BLE001 — 실행 하나가 넘어져도 서비스는 산다
            run.steps = [Step(1, "assess", "failed", f"실행 중 오류: {error}")]
            run.status = "failed"
            return

        self._finish(run, plan, {profile.age_group for profile in profiles})

    def _finish(self, run: Run, plan: Plan, age_groups: set[str]) -> None:
        run.steps = plan.steps
        if plan.refused or plan.proposal is None:
            run.status = "refused"
            run.refused = True
            run.refusal_reason = plan.refusal_reason
            return

        problems = verify.check_proposal(plan.proposal, age_groups)
        if problems:
            run.steps = [*plan.steps[:3], Step(4, "verify", "failed", " · ".join(problems))]
            run.status = "refused"
            run.refused = True
            run.refusal_reason = "no_citation_generated"
            return

        citations = len(plan.proposal.get("citations") or [])
        run.steps = [
            *plan.steps[:3],
            Step(4, "verify", "ok", f"인용 {citations}건 · 금지 어휘 0건"),
        ]
        run.status = "succeeded"
        run.proposal = plan.proposal

    def get(self, run_id: str) -> Run:
        run = self._runs.get(run_id)
        if run is None:
            raise run_not_found(run_id)
        return run


store = Store()
