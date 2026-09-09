"""등급 판정 (docs/dev/AI-2).

**판정 규칙의 정본은 공단이다** — 국민체력100 인증단계 안내와 `등급평가항목및기준`
시트를 대조해 옮겼다. `../../docs/02` §5.2 의 서술은 이보다 거칠다 (docs/dev/AI-2 §6).

| 등급 | 조건 |
|---|---|
| 1등급 | 건강체력이 **모두** 상위 30% · 운동체력 **중 한 가지**가 상위 30% |
| 2등급 | 건강체력이 **모두** 상위 50% · 운동체력 **중 한 가지**가 상위 50% |
| 3등급 | 건강체력이 **모두** 상위 70% · 신체조성 **중 한 가지**가 권장 범위 |
| 4등급 | 심폐지구력 **과** 근력이 3등급 기준 이상 |
| 5등급 | 심폐지구력 **또는** 근력이 3등급 기준 이상 |
| 6등급 | 5등급 미달 |

문턱 표의 1·2·3등급 값이 각각 상위 30·50·70% 컷이다.

**운동체력은 "중 한 가지"다.** 전부 요구하면 성인 반응시간처럼 문턱이 빡빡한 항목
하나가 상위 등급을 통째로 막는다. 이것이 판정이 공단 기록보다 박했던 이유다.

**신체조성은 3등급에만 쓴다.** 시트에서 1·2등급 행의 BMI·체지방률 칸은 비어 있다.

**등급과 요인 점수는 다른 것이다.** 점수는 항목 하나를 문턱에 맞춘 0~100 연속값이고
(`score.py`), 등급은 항목 조합 조건이다. 둘을 한 함수에서 내지 않는다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from .criteria import BodyRange, Threshold
from .items import ITEMS

HEALTH_FACTORS = ("심폐지구력", "근력", "근지구력", "유연성")
SKILL_FACTORS = ("민첩성", "순발력", "협응력")

# 4·5·6등급은 2025-06 개편으로 생겼고 청소년·성인·어르신에만 있다. 유아기·유소년은
# 3등급 아래가 `참가` 다 — 원자료 전수에서 확인했다 (docs/dev/AI-2 §5.3).
EXTENDED_AGE_GROUPS = ("청소년", "성인", "어르신")

GRADE_NAMES = {1: "1등급", 2: "2등급", 3: "3등급", 4: "4등급", 5: "5등급", 6: "6등급"}
PARTICIPATED = "참가"

# 신체조성. 점수화하지 않지만 3등급 판정에는 쓴다 (docs/02 §5.2 vs §5.4).
# 문턱이 구간 형태라 아직 문턱 표에 없다 (docs/dev/AI-2 §6).
BODY_COMPOSITION = frozenset({"003", "004", "018", "042"})


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNDECIDABLE = "undecidable"  # 판정 항목이 측정되지 않았다


@dataclass(frozen=True)
class GradeCheck:
    """한 등급의 판정 결과. 왜 그렇게 됐는지를 함께 남긴다."""

    grade: int
    verdict: Verdict
    missing_factors: tuple[str, ...] = ()
    failed_factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class GradeResult:
    """`parent_scope.grade` 와 그 근거 (docs/03 §3.4)."""

    grade: str | None
    checks: tuple[GradeCheck, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def summary(self) -> str:
        """`steps[].summary` 에 실을 한 줄. 사람이 읽는 감사 기록이다 (docs/03 §5.4)."""
        if self.grade is None:
            missing = sorted({f for c in self.checks for f in c.missing_factors})
            return f"등급 판정 불가 · 측정되지 않은 요인 {', '.join(missing) or '없음'}"

        awarded = next((g for g, name in GRADE_NAMES.items() if name == self.grade), 99)
        above = [c for c in self.checks if c.grade < awarded]
        parts = [f"등급 {self.grade}"]

        blocked = next((c for c in above if c.verdict is Verdict.FAIL and c.failed_factors), None)
        if blocked is not None:
            parts.append(
                f"{GRADE_NAMES[blocked.grade]} 문턱에 걸린 요인 {', '.join(blocked.failed_factors)}"
            )
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


def _by_factor(cell: list[Threshold], grade: int) -> dict[str, list[Threshold]]:
    out: dict[str, list[Threshold]] = defaultdict(list)
    for t in cell:
        if t.grade == grade and t.item_code in ITEMS:
            out[ITEMS[t.item_code].factor].append(t)
    return out


def _factor_state(candidates: list[Threshold], measurements: dict[str, float]) -> bool | None:
    """한 요인의 통과 여부. 측정되지 않았으면 `None`.

    기준항목이 둘인 요인(심폐지구력 020/035/037, 성인 민첩성 021/040 등)은 택1이다.
    잰 것만 보고, 둘 이상 쟀으면 둘 다 넘어야 한다 — 높은 쪽을 고르는 것은
    docs/02 §5.4 가 금지한 "최고점 고르기"다.
    """
    measured = [t for t in candidates if t.item_code in measurements]
    if not measured:
        return None
    return all(
        _passes(
            measurements[t.item_code], t.value, lower_is_better=ITEMS[t.item_code].lower_is_better
        )
        for t in measured
    )


def _check_upper(grade: int, cell: list[Threshold], measurements: dict[str, float]) -> GradeCheck:
    """1·2등급 — 건강체력은 모두, 운동체력은 중 한 가지."""
    by = _by_factor(cell, grade)
    missing: list[str] = []
    failed: list[str] = []

    for factor in HEALTH_FACTORS:
        if factor not in by:
            continue  # 그 연령대가 재지 않는 요인이다
        state = _factor_state(by[factor], measurements)
        if state is None:
            missing.append(factor)
        elif not state:
            failed.append(factor)

    skill = {f: by[f] for f in SKILL_FACTORS if f in by}
    if skill:
        states = {f: _factor_state(ts, measurements) for f, ts in skill.items()}
        if any(s is True for s in states.values()):
            pass  # 한 가지가 넘었다
        elif all(s is None for s in states.values()):
            missing.append("운동체력")
        else:
            failed.append("운동체력")

    if missing:
        return GradeCheck(grade, Verdict.UNDECIDABLE, tuple(missing), tuple(failed))
    if failed:
        return GradeCheck(grade, Verdict.FAIL, (), tuple(failed))
    if not by:
        return GradeCheck(grade, Verdict.UNDECIDABLE, ("문턱 없음",), ())
    return GradeCheck(grade, Verdict.PASS)


def _check_third(
    cell: list[Threshold], measurements: dict[str, float], body: list[BodyRange]
) -> GradeCheck:
    """3등급 — 건강체력은 모두, 운동체력은 보지 않는다, 신체조성은 권장 범위."""
    by = _by_factor(cell, 3)
    if not by:
        return GradeCheck(3, Verdict.UNDECIDABLE, ("문턱 없음",), ())
    missing: list[str] = []
    failed: list[str] = []
    for factor in HEALTH_FACTORS:
        if factor not in by:
            continue
        state = _factor_state(by[factor], measurements)
        if state is None:
            missing.append(factor)
        elif not state:
            failed.append(factor)
    # 신체조성은 3등급에만 본다. 시트에서 1·2등급 행의 BMI·체지방률 칸은 비어 있다.
    # **중 한 가지**가 권장 범위면 통과다 — 운동체력과 같은 형태이고, 전부 요구하면
    # 공단 기록보다 크게 박해진다 (docs/dev/AI-2 §5.4).
    measured_body = [r for r in body if r.item_code in measurements]
    if body and not measured_body:
        missing.append("신체조성")
    elif measured_body and not any(r.contains(measurements[r.item_code]) for r in measured_body):
        failed.append("신체조성")

    if missing:
        return GradeCheck(3, Verdict.UNDECIDABLE, tuple(missing), tuple(failed))
    return GradeCheck(3, Verdict.FAIL if failed else Verdict.PASS, (), tuple(failed))


def _check_lower(grade: int, cell: list[Threshold], measurements: dict[str, float]) -> GradeCheck:
    """4·5등급 — 심폐지구력과 근력을 3등급 기준으로 본다. 4는 AND, 5는 OR."""
    by = _by_factor(cell, 3)
    states = {f: _factor_state(by[f], measurements) for f in ("심폐지구력", "근력") if f in by}
    if not states or all(s is None for s in states.values()):
        return GradeCheck(grade, Verdict.UNDECIDABLE, ("심폐지구력", "근력"), ())
    passed = [f for f, s in states.items() if s is True]
    ok = len(passed) == len(states) if grade == 4 else bool(passed)
    if ok:
        return GradeCheck(grade, Verdict.PASS)
    return GradeCheck(grade, Verdict.FAIL, (), tuple(f for f, s in states.items() if s is not True))


def judge(
    thresholds: list[Threshold],
    *,
    age_group: str,
    age: int,
    sex: str,
    measurements: dict[str, float],
    body_ranges: list[BodyRange] | None = None,
) -> GradeResult:
    """위에서부터 내려가며 처음 만족하는 등급이 답이다.

    **측정되지 않아 판정할 수 없는 것과 기준을 못 넘은 것은 다르다.** 결측을 통과로
    보면 적게 잰 사람이 높은 등급을 받고, 결측을 미달로 보면 안 잰 사람이 바닥에
    깔린다. 아래까지 내려가도 판정이 안 되면 `None` 이다.
    """
    cell = band_thresholds(thresholds, age_group=age_group, age=age, sex=sex)
    if not cell:
        return GradeResult(grade=None, notes=("그 연령 구간의 기준표가 없다",))

    body = [
        r
        for r in (body_ranges or [])
        if r.age_group == age_group and r.sex == sex and r.age_lo <= age <= r.age_hi
    ]
    extended = age_group in EXTENDED_AGE_GROUPS
    checks = [_check_upper(1, cell, measurements), _check_upper(2, cell, measurements)]
    checks.append(_check_third(cell, measurements, body))
    if extended:
        checks += [_check_lower(4, cell, measurements), _check_lower(5, cell, measurements)]
    frozen = tuple(checks)
    notes = _notes(body)

    for check in frozen:
        if check.verdict is Verdict.PASS:
            return GradeResult(GRADE_NAMES[check.grade], frozen, notes)

    # 바닥 등급은 연령대에 따라 다르다. 유아기·유소년에는 4~6등급이 없다.
    floor = frozen[-1]
    if floor.verdict is Verdict.FAIL:
        return GradeResult(GRADE_NAMES[6] if extended else PARTICIPATED, frozen, notes)
    return GradeResult(None, frozen, notes)


def _notes(body: list[BodyRange]) -> tuple[str, ...]:
    """판정이 무엇을 보지 못했는지 남긴다. 조용히 빠진 것은 나중에 결함으로 읽힌다."""
    if body:
        return ()
    return ("신체조성 구간이 없어 3등급 판정에서 빠졌다 (docs/dev/AI-2 §6)",)
