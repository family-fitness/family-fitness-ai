"""등급 판정 (docs/dev/AI-2).

판정 규칙의 정본은 공단이다 — 인증단계 안내와 `등급평가항목및기준` 시트를 대조해
옮겼다. 검사는 그 규칙을 문장 그대로 확인한다.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from family_fitness_ai.stats.criteria import BodyRange, Threshold, load, parse_body_range
from family_fitness_ai.stats.grade import PARTICIPATED, Verdict, judge

CRITERIA = pathlib.Path("data/release/grade_thresholds.csv")


@pytest.fixture(scope="module")
def thresholds() -> list[Threshold]:
    if not CRITERIA.exists():
        pytest.skip("문턱 표가 없다")
    return load(CRITERIA)


def t(code: str, grade: int, value: float, *, age_group: str = "유소년") -> Threshold:
    return Threshold(age_group, "F", 11, 11, code, grade, value)


def cell_for(age_group: str) -> list[Threshold]:
    """건강체력 넷 + 운동체력 둘. 운동체력에는 3등급 문턱이 없다."""
    return [
        *(t("028", g, v, age_group=age_group) for g, v in ((1, 44.4), (2, 39.5), (3, 34.8))),
        *(t("012", g, v, age_group=age_group) for g, v in ((1, 10.9), (2, 6.5), (3, 3.0))),
        *(t("020", g, v, age_group=age_group) for g, v in ((1, 62), (2, 51), (3, 40))),
        *(t("009", g, v, age_group=age_group) for g, v in ((1, 40), (2, 30), (3, 20))),
        *(t("022", g, v, age_group=age_group) for g, v in ((1, 165), (2, 146))),  # 순발력
        *(t("043", g, v, age_group=age_group) for g, v in ((1, 50), (2, 40))),  # 민첩성
    ]


CELL = cell_for("유소년")
HEALTHY = {"028": 50.0, "012": 12.0, "020": 70, "009": 45}


def g(measurements: dict[str, float], *, age_group: str = "유소년", body=None) -> str | None:
    return judge(
        cell_for(age_group),
        age_group=age_group,
        age=11,
        sex="F",
        measurements=measurements,
        body_ranges=body,
    ).grade


# ── 1·2등급 — 건강체력은 모두, 운동체력은 중 한 가지 ────────────────────


def test_운동체력은_중_한_가지만_넘으면_된다() -> None:
    """전부 요구하면 문턱이 빡빡한 항목 하나가 상위 등급을 통째로 막는다."""
    # 순발력은 1등급, 민첩성은 미달 — 그래도 1등급이다
    assert g({**HEALTHY, "022": 170, "043": 10}) == "1등급"


def test_운동체력이_전부_미달이면_등급이_내려간다() -> None:
    assert g({**HEALTHY, "022": 50, "043": 10}) == "3등급"


def test_건강체력은_하나만_걸려도_내려간다() -> None:
    """건강체력은 AND 조건이다."""
    assert g({**HEALTHY, "012": 7.0, "022": 170}) == "2등급"


def test_운동체력을_하나도_재지_않으면_상위_등급은_판정_불가다() -> None:
    result = judge(CELL, age_group="유소년", age=11, sex="F", measurements=HEALTHY)
    check1 = next(c for c in result.checks if c.grade == 1)
    assert check1.verdict is Verdict.UNDECIDABLE
    assert "운동체력" in check1.missing_factors
    assert result.grade == "3등급"  # 3등급은 운동체력을 보지 않으므로 판정된다


# ── 3등급 — 건강체력 + 신체조성. 운동체력 3요인은 반영하지 않는다 ────────


def test_3등급은_운동체력을_보지_않는다() -> None:
    """시트의 3등급 행에서 운동체력 칸은 비어 있다."""
    result = judge(
        CELL, age_group="유소년", age=11, sex="F", measurements={**HEALTHY, "022": 1, "043": 1}
    )
    check3 = next(c for c in result.checks if c.grade == 3)
    assert check3.verdict is Verdict.PASS
    assert result.grade == "3등급"


def test_신체조성은_중_한_가지가_범위에_들면_통과다() -> None:
    body = [
        BodyRange("유소년", "F", 11, 11, "018", None, 23.3),
        BodyRange("유소년", "F", 11, 11, "042", None, 0.47),
    ]
    ok = {**HEALTHY, "018": 30.0, "042": 0.40}  # BMI 는 벗어나고 WHtR 는 든다
    assert g(ok, body=body) == "3등급"


def test_신체조성이_전부_벗어나면_3등급이_아니다() -> None:
    body = [BodyRange("유소년", "F", 11, 11, "018", None, 23.3)]
    assert g({**HEALTHY, "018": 30.0}, body=body) == PARTICIPATED


def test_신체조성_구간이_없으면_그_사실을_남긴다() -> None:
    result = judge(CELL, age_group="유소년", age=11, sex="F", measurements=HEALTHY)
    assert any("신체조성" in n for n in result.notes)


# ── 3등급 미달은 참가다. 4·5·6등급은 내지 않는다 ─────────────────────


def test_3등급에_못_미치면_참가다() -> None:
    assert g({"028": 1.0, "020": 5, "012": 0.0, "009": 1}) == PARTICIPATED


def test_성인도_참가까지만_낸다() -> None:
    """공단은 3등급 아래를 4·5·6으로 쪼개지만 계약의 grade 는 넷이다."""
    assert g({"028": 1.0, "020": 5, "012": 0.0, "009": 1}, age_group="성인") == PARTICIPATED


# ── 결측 ────────────────────────────────────────────────────────────


def test_결측이_통과로_새지_않는다() -> None:
    """적게 잰 사람이 높은 등급을 받으면 안 된다."""
    result = judge(CELL, age_group="유소년", age=11, sex="F", measurements={"028": 50.0})
    assert result.grade is None


def test_판정_불가와_참가는_다르다() -> None:
    """기준을 못 넘은 것과 재지 않은 것은 다르다."""
    assert g({"028": 1.0, "012": 0.0, "020": 5, "009": 1}) == PARTICIPATED
    assert g({"028": 1.0, "012": 0.0}) is None


def test_연령_구간_밖이면_판정하지_않는다(thresholds: list[Threshold]) -> None:
    """만 7~10세는 측정 0건이라 기준표에도 없다 (docs/02 §2.2)."""
    assert (
        judge(thresholds, age_group="유소년", age=8, sex="F", measurements={"028": 40.0}).grade
        is None
    )


# ── 방향과 감사 기록 ─────────────────────────────────────────────────


def test_작을수록_우수한_항목은_부등호가_뒤집힌다() -> None:
    cell = [t("021", g_, v, age_group="성인") for g_, v in ((1, 12.0), (2, 13.0))]
    cell += [t("028", g_, v, age_group="성인") for g_, v in ((1, 44.4), (2, 39.5), (3, 34.8))]
    fast = judge(cell, age_group="성인", age=11, sex="F", measurements={"021": 11.0, "028": 50.0})
    slow = judge(cell, age_group="성인", age=11, sex="F", measurements={"021": 14.0, "028": 50.0})
    assert fast.grade == "1등급"
    assert slow.grade != "1등급"


def test_summary_는_왜_그_등급인지_적는다() -> None:
    result = judge(
        CELL, age_group="유소년", age=11, sex="F", measurements={**HEALTHY, "012": 7.0, "022": 170}
    )
    assert "유연성" in result.summary()


def test_상위_등급이_판정_불가면_요약이_그것을_말한다() -> None:
    result = judge(CELL, age_group="유소년", age=11, sex="F", measurements=HEALTHY)
    assert "상위 등급은 판정 불가" in result.summary()
    assert "운동체력" in result.summary()


# ── 신체조성 구간 파싱 ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("18.5이상 25미만", (18.5, 25.0)),
        ("7%초과 27%미만", (7.0, 27.0)),
        ("< 24.2", (None, 24.2)),
        ("<17.5", (None, 17.5)),
        ("< .50", (None, 0.5)),
        ("62", None),
        ("", None),
    ],
)
def test_신체조성_구간을_읽는다(text: str, expected: tuple | None) -> None:
    assert parse_body_range(text) == expected


def test_구간은_lo_이상_hi_미만이다() -> None:
    r = BodyRange("성인", "M", 19, 24, "018", 18.5, 25.0)
    assert r.contains(18.5) and r.contains(24.9)
    assert not r.contains(18.4) and not r.contains(25.0)


# ── 또래 등급 분포 ───────────────────────────────────────────────────


def frame(grades: list[str | None], *, ym: str = "202607") -> pd.DataFrame:
    from family_fitness_ai.ingest import measurements as M

    return pd.DataFrame(
        {
            M.AGE_GROUP_COL: ["유소년"] * len(grades),
            M.AGE_COL: [11] * len(grades),
            M.SEX_COL: ["F"] * len(grades),
            M.DATE_COL: [f"{ym}01"] * len(grades),
            M.GRADE_COL: grades,
        }
    )


def test_등급_미기재는_분모에서도_뺀다() -> None:
    """미판정을 참가로 세면 참가 비율이 부풀어 오른다."""
    from family_fitness_ai.stats.distribution import build_grade_distribution

    out = build_grade_distribution(
        frame(["1등급", "3등급", "참가", None, "미인증"]), [t("028", 1, 44.4)]
    )
    assert out["n_cell"].unique().tolist() == [3]
    assert out["count"].sum() == 3


def test_등급_분포는_판정을_다시_돌리지_않는다() -> None:
    """원자료의 등급 컬럼을 센다 — 측정값이 하나도 없어도 분포는 나온다."""
    from family_fitness_ai.stats.distribution import build_grade_distribution

    out = build_grade_distribution(frame(["1등급", "참가"]), [t("028", 1, 44.4)])
    assert set(out["grade"]) == {"1등급", "2등급", "3등급", "참가"}
    assert out.loc[out.grade == "1등급", "ratio"].item() == pytest.approx(0.5)


def test_4에서_6등급은_참가로_접는다() -> None:
    """셋 다 3등급 미달이라 뜻이 바뀌지 않는다. 접으면 개편 전후가 같은 눈금이다."""
    from family_fitness_ai.stats.distribution import build_grade_distribution, fold_grades

    assert list(fold_grades(pd.Series(["4등급", "5등급", "6등급", "3등급"]))) == [
        "참가",
        "참가",
        "참가",
        "3등급",
    ]
    out = build_grade_distribution(frame(["4등급", "5등급", "6등급", "1등급"]), [t("028", 1, 44.4)])
    assert out.loc[out.grade == "참가", "count"].item() == 3
    assert out["n_cell"].unique().tolist() == [4]


def test_개편_전_행도_분포에_넣는다() -> None:
    """접고 나면 두 제도가 같은 눈금이라 기간을 자를 이유가 없다."""
    from family_fitness_ai.stats.distribution import build_grade_distribution

    old = build_grade_distribution(frame(["1등급", "참가"], ym="202501"), [t("028", 1, 44.4)])
    assert old["n_cell"].unique().tolist() == [2]


def test_항목군은_개별_요인과_섞어_적지_않는다() -> None:
    """`운동체력`·`신체조성` 은 요인이 아니라 '중 한 가지' 조건이 묶인 항목군이다."""
    body = [BodyRange("유소년", "F", 11, 11, "018", None, 23.3)]
    blocked = judge(
        CELL,
        age_group="유소년",
        age=11,
        sex="F",
        measurements={**HEALTHY, "012": 7.0, "022": 50, "043": 10, "018": 18.0},
        body_ranges=body,
    )
    line = blocked.summary()
    assert "요인 " in line
    assert "요인 근력, 근지구력, 유연성, 운동체력" not in line  # 한 줄에 섞이지 않는다
    assert line.rstrip().endswith("운동체력")


def test_결측_문구도_같은_규칙을_따른다() -> None:
    result = judge(CELL, age_group="유소년", age=11, sex="F", measurements={"028": 50.0})
    assert "측정되지 않은 요인" in result.summary()
