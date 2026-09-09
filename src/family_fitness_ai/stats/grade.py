"""등급 판정 (docs/02 §5.2 · docs/dev/AI-2).

**등급과 요인 점수는 다른 것이다.** 점수는 항목 하나를 문턱에 맞춘 0~100 연속값이고
(`score.py`), 등급은 여러 항목의 AND 조건이다. 요인 점수가 전부 높아도 한 항목이
문턱에 걸리면 등급은 내려간다. 둘을 한 함수에서 내지 않는다.

**등급마다 보는 항목이 다르다.** 3등급은 건강체력만 보고, 1·2등급은 운동체력까지
본다. 그 구분을 코드에 다시 적지 않는다 — 기준표가 이미 그렇게 생겼다. 운동체력
항목에는 3등급 문턱이 아예 없으므로 "그 등급에 문턱이 있는 항목"을 모으면 구분이
그대로 나온다 (docs/02 §5.2).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from .criteria import Threshold
from .items import ITEMS

GRADE_NAMES = {1: "1등급", 2: "2등급", 3: "3등급"}
PARTICIPATED = "참가"


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNDECIDABLE = "undecidable"  # 판정 항목이 측정되지 않았다


@dataclass(frozen=True)
class GradeCheck:
    """한 등급의 판정 결과. 왜 그렇게 됐는지를 함께 남긴다."""

    grade: int
    verdict: Verdict
    required_factors: tuple[str, ...] = ()
    missing_factors: tuple[str, ...] = ()
    failed_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class GradeResult:
    """`parent_scope.grade` 와 그 근거 (docs/03 §3.4)."""

    grade: str | None
    checks: tuple[GradeCheck, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def summary(self) -> str:
        """`steps[].summary` 에 실을 한 줄. 사람이 읽는 감사 기록이다 (docs/03 §5.4).

        **받은 등급만 적으면 왜 그 등급인지 감춘다.** 상위 등급이 문턱에 걸려
        내려온 것과, 상위 등급을 판정할 재료가 없어 내려온 것은 다르다.
        """
        if self.grade is None:
            missing = sorted({f for c in self.checks for f in c.missing_factors})
            return f"등급 판정 불가 · 측정되지 않은 요인 {', '.join(missing) or '없음'}"

        awarded = next((g for g, name in GRADE_NAMES.items() if name == self.grade), 4)
        above = [c for c in self.checks if c.grade < awarded]
        parts = [f"등급 {self.grade}"]

        blocked = next((c for c in above if c.verdict is Verdict.FAIL and c.failed_items), None)
        if blocked is not None:
            names = ", ".join(ITEMS[c].name for c in blocked.failed_items if c in ITEMS)
            parts.append(f"{GRADE_NAMES[blocked.grade]} 문턱에 걸린 항목 {names}")

        undecided = sorted(
            {f for c in above if c.verdict is Verdict.UNDECIDABLE for f in c.missing_factors}
        )
        if undecided:
            parts.append(f"상위 등급은 판정 불가 · 측정되지 않은 요인 {', '.join(undecided)}")
        return " · ".join(parts)


def band_thresholds(
    thresholds: list[Threshold], *, age_group: str, age: int, sex: str
) -> list[Threshold]:
    """그 사람이 속한 연령 구간의 문턱만 고른다. 유아기는 개월이다 (docs/02 §2.4)."""
    return [
        t
        for t in thresholds
        if t.age_group == age_group and t.sex == sex and t.age_lo <= age <= t.age_hi
    ]


def _passes(value: float, threshold: float, *, lower_is_better: bool) -> bool:
    """방향의 정본은 `items.Item.lower_is_better` 다 (docs/02 §5.3)."""
    return value <= threshold if lower_is_better else value >= threshold


def _check(grade: int, cell: list[Threshold], measurements: dict[str, float]) -> GradeCheck:
    """한 등급을 판정한다. 요인 단위로 본다 — 기준항목이 둘인 요인이 있다.

    택1 측정(심폐지구력 020/035/037 등)에서 재지 않은 쪽을 결측으로 보면 정상인
    사람이 판정 불가가 된다. 잰 것만 보되, 둘 이상 쟀으면 **둘 다** 통과해야 한다 —
    더 높은 쪽을 고르는 것은 docs/02 §5.4 가 금지한 "최고점 고르기"다.
    """
    by_factor: dict[str, list[Threshold]] = defaultdict(list)
    for t in cell:
        if t.grade == grade and t.item_code in ITEMS:
            by_factor[ITEMS[t.item_code].factor].append(t)

    missing: list[str] = []
    failed: list[str] = []
    for factor, candidates in by_factor.items():
        measured = [t for t in candidates if t.item_code in measurements]
        if not measured:
            missing.append(factor)
            continue
        failed += [
            t.item_code
            for t in measured
            if not _passes(
                measurements[t.item_code],
                t.value,
                lower_is_better=ITEMS[t.item_code].lower_is_better,
            )
        ]

    if not by_factor:
        # 그 등급의 문턱이 이 칸에 하나도 없다. 조건이 공집합이라고 통과시키면
        # 기준이 없는 등급을 모두가 받는다 — 판정하지 않는 것이 맞다.
        verdict = Verdict.UNDECIDABLE
    elif missing:
        verdict = Verdict.UNDECIDABLE
    elif failed:
        verdict = Verdict.FAIL
    else:
        verdict = Verdict.PASS
    return GradeCheck(
        grade=grade,
        verdict=verdict,
        required_factors=tuple(sorted(by_factor)),
        missing_factors=tuple(sorted(missing)),
        failed_items=tuple(sorted(set(failed))),
    )


def judge(
    thresholds: list[Threshold],
    *,
    age_group: str,
    age: int,
    sex: str,
    measurements: dict[str, float],
) -> GradeResult:
    """위에서부터 내려가며 처음 만족하는 등급이 답이다 (docs/02 §5.2).

    **측정되지 않은 판정 항목이 있으면 그 등급은 판정하지 않는다.** 결측을 통과로
    보면 적게 잰 사람이 높은 등급을 받는다. 3등급까지 내려가도 판정이 안 되면
    `참가` 가 아니라 `None` 이다 — 기준을 못 넘은 것과 재지 않은 것은 다르다.
    """
    cell = band_thresholds(thresholds, age_group=age_group, age=age, sex=sex)
    if not cell:
        return GradeResult(grade=None, notes=("그 연령 구간의 기준표가 없다",))

    checks = tuple(_check(g, cell, measurements) for g in (1, 2, 3))
    notes = _notes(cell)

    for check in checks:
        if check.verdict is Verdict.PASS:
            return GradeResult(grade=GRADE_NAMES[check.grade], checks=checks, notes=notes)

    lowest = checks[-1]
    if lowest.verdict is Verdict.FAIL:
        return GradeResult(grade=PARTICIPATED, checks=checks, notes=notes)
    return GradeResult(grade=None, checks=checks, notes=notes)


def _notes(cell: list[Threshold]) -> tuple[str, ...]:
    """판정이 무엇을 보지 못했는지 남긴다. 조용히 빠진 것은 나중에 결함으로 읽힌다."""
    notes: list[str] = []
    if not any(t.item_code in BODY_COMPOSITION for t in cell):
        # docs/dev/AI-2 §5 — 기준표의 신체조성 칸이 구간 형태라 문턱 표에 없다.
        # 빠진 채로 판정하면 등급이 실제보다 후하다.
        notes.append("신체조성 문턱이 없어 판정에서 빠졌다 (docs/dev/AI-2 §5)")
    return tuple(notes)


# 신체조성. 점수화하지 않지만 등급 판정에는 쓴다 (docs/02 §5.2 vs §5.4).
BODY_COMPOSITION = frozenset({"003", "004", "018", "042"})
