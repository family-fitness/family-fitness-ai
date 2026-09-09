"""등급 판정 (docs/dev/AI-2)."""

from __future__ import annotations

import pathlib

import pytest

from family_fitness_ai.stats.criteria import Threshold, load
from family_fitness_ai.stats.grade import (
    GRADE_NAMES,
    PARTICIPATED,
    Verdict,
    judge,
)

CRITERIA = pathlib.Path("data/release/grade_thresholds.csv")


@pytest.fixture(scope="module")
def thresholds() -> list[Threshold]:
    if not CRITERIA.exists():
        pytest.skip("문턱 표가 없다")
    return load(CRITERIA)


def t(code: str, grade: int, value: float, *, age_group: str = "유소년") -> Threshold:
    return Threshold(age_group, "F", 11, 11, code, grade, value)


# 유소년 여 11세를 본뜬 최소 문턱. 건강체력 넷 + 운동체력 하나.
CELL = [
    *(t("028", g, v) for g, v in ((1, 44.4), (2, 39.5), (3, 34.8))),  # 근력
    *(t("012", g, v) for g, v in ((1, 10.9), (2, 6.5), (3, 3.0))),  # 유연성
    *(t("020", g, v) for g, v in ((1, 62), (2, 51), (3, 40))),  # 심폐지구력
    *(t("009", g, v) for g, v in ((1, 40), (2, 30), (3, 20))),  # 근지구력
    *(t("022", g, v) for g, v in ((1, 165), (2, 146))),  # 순발력 — 3등급 문턱이 없다
]
FULL = {"028": 50.0, "012": 12.0, "020": 70, "009": 45, "022": 170}


def judge_cell(measurements: dict[str, float]) -> object:
    return judge(CELL, age_group="유소년", age=11, sex="F", measurements=measurements)


def test_전부_1등급_문턱을_넘으면_1등급이다() -> None:
    assert judge_cell(FULL).grade == GRADE_NAMES[1]


def test_한_항목이_걸리면_등급이_내려간다() -> None:
    """요인 점수가 아무리 높아도 AND 조건이다 (docs/02 §5.2)."""
    result = judge_cell({**FULL, "012": 7.0})  # 유연성만 2등급 구간
    assert result.grade == GRADE_NAMES[2]


def test_3등급은_운동체력을_보지_않는다() -> None:
    """운동체력에 3등급 문턱이 없는 것은 결측이 아니다 (docs/02 §5.2)."""
    result = judge(
        CELL,
        age_group="유소년",
        age=11,
        sex="F",
        measurements={"028": 35.0, "012": 3.5, "020": 41, "009": 21},  # 순발력 미측정
    )
    assert result.grade == GRADE_NAMES[3]
    check3 = next(c for c in result.checks if c.grade == 3)
    assert "순발력" not in check3.required_factors
    assert check3.verdict is Verdict.PASS


def test_1등급_판정은_순발력이_없으면_판정_불가다() -> None:
    result = judge(
        CELL,
        age_group="유소년",
        age=11,
        sex="F",
        measurements={"028": 50.0, "012": 12.0, "020": 70, "009": 45},
    )
    check1 = next(c for c in result.checks if c.grade == 1)
    assert check1.verdict is Verdict.UNDECIDABLE
    assert check1.missing_factors == ("순발력",)
    assert result.grade == GRADE_NAMES[3]  # 3등급까지는 판정된다


def test_결측이_통과로_새지_않는다() -> None:
    """적게 잰 사람이 높은 등급을 받으면 안 된다."""
    result = judge_cell({"028": 50.0})
    assert result.grade is None
    assert all(c.verdict is Verdict.UNDECIDABLE for c in result.checks)


def test_다_재고_못_넘으면_참가다() -> None:
    result = judge_cell({"028": 10.0, "012": 0.0, "020": 5, "009": 1, "022": 50})
    assert result.grade == PARTICIPATED


def test_판정_불가와_참가는_다르다() -> None:
    """기준을 못 넘은 것과 재지 않은 것은 다르다."""
    참가 = judge_cell({"028": 10.0, "012": 0.0, "020": 5, "009": 1, "022": 50})
    불가 = judge_cell({"028": 10.0, "012": 0.0})
    assert 참가.grade == PARTICIPATED
    assert 불가.grade is None


def test_작을수록_우수한_항목은_부등호가_뒤집힌다() -> None:
    # 작을수록 우수하므로 문턱은 등급이 낮을수록 커진다.
    cell = [t("021", g, v, age_group="성인") for g, v in ((1, 12.0), (2, 13.0), (3, 14.0))]
    fast = judge(cell, age_group="성인", age=11, sex="F", measurements={"021": 11.0})
    slow = judge(cell, age_group="성인", age=11, sex="F", measurements={"021": 14.5})
    assert fast.grade == GRADE_NAMES[1]
    assert slow.grade == PARTICIPATED


def test_택1_항목은_잰_것만_본다() -> None:
    """심폐지구력 020/035/037 은 택1이다 — 안 잰 쪽이 결측이 아니다 (docs/02 §5.4)."""
    cell = [
        t("020", 1, 62),
        t("035", 1, 45.0),
        t("037", 1, 45.0),
        t("028", 1, 44.4),
    ]
    result = judge(cell, age_group="유소년", age=11, sex="F", measurements={"020": 70, "028": 50.0})
    assert result.grade == GRADE_NAMES[1]


def test_택1을_둘_이상_재면_둘_다_넘어야_한다() -> None:
    """더 높은 쪽을 고르는 것은 최고점 고르기다 (docs/02 §5.4)."""
    cell = [t("020", 1, 62), t("035", 1, 45.0), t("028", 1, 44.4)]
    result = judge(
        cell,
        age_group="유소년",
        age=11,
        sex="F",
        measurements={"020": 70, "035": 30.0, "028": 50.0},
    )
    check1 = next(c for c in result.checks if c.grade == 1)
    assert check1.verdict is Verdict.FAIL
    assert "035" in check1.failed_items  # 020 을 넘었다고 덮이지 않는다


def test_신체조성이_빠진_것을_조용히_넘기지_않는다() -> None:
    """docs/dev/AI-2 §5 — 빠진 채로 판정하면 등급이 실제보다 후하다."""
    result = judge_cell(FULL)
    assert any("신체조성" in n for n in result.notes)


def test_summary_는_왜_그_등급인지_적는다() -> None:
    result = judge_cell({**FULL, "012": 7.0})
    assert "앉아윗몸앞으로굽히기" in result.summary()


def test_그_등급의_문턱이_없으면_통과가_아니라_판정_불가다() -> None:
    """조건이 공집합이라고 통과시키면 기준이 없는 등급을 모두가 받는다."""
    only_grade_one = [t("028", 1, 44.4)]
    result = judge(only_grade_one, age_group="유소년", age=11, sex="F", measurements={"028": 50.0})
    assert result.grade == GRADE_NAMES[1]
    assert next(c for c in result.checks if c.grade == 3).verdict is Verdict.UNDECIDABLE


def test_연령_구간_밖이면_판정하지_않는다(thresholds: list[Threshold]) -> None:
    """만 7~10세는 측정 0건이라 기준표에도 없다 (docs/02 §2.2)."""
    result = judge(thresholds, age_group="유소년", age=8, sex="F", measurements={"028": 40.0})
    assert result.grade is None


def test_기준표에는_3등급_문턱이_건강체력에만_있다(thresholds: list[Threshold]) -> None:
    """판정 항목 구분을 코드에 다시 적지 않는 근거다 (docs/02 §5.2)."""
    from family_fitness_ai.stats.items import ITEMS

    운동체력 = {"순발력", "민첩성", "협응력"}
    factors_with_grade3 = {
        ITEMS[t.item_code].factor for t in thresholds if t.grade == 3 and t.item_code in ITEMS
    }
    assert not (factors_with_grade3 & 운동체력)


def test_상위_등급이_판정_불가면_요약이_그것을_말한다() -> None:
    """3등급을 받았는데 왜 그 위가 아닌지가 감사 기록에 남아야 한다."""
    result = judge(
        CELL,
        age_group="유소년",
        age=11,
        sex="F",
        measurements={"028": 50.0, "012": 12.0, "020": 70, "009": 45},  # 순발력 미측정
    )
    assert result.grade == GRADE_NAMES[3]
    assert "상위 등급은 판정 불가" in result.summary()
    assert "순발력" in result.summary()


def test_문턱에_걸려_내려온_것과_재료가_없어_내려온_것을_가른다() -> None:
    걸림 = judge_cell({**FULL, "012": 7.0})
    없음 = judge(
        CELL,
        age_group="유소년",
        age=11,
        sex="F",
        measurements={"028": 50.0, "012": 12.0, "020": 70, "009": 45},
    )
    assert "문턱에 걸린 항목" in 걸림.summary()
    assert "문턱에 걸린 항목" not in 없음.summary()
