"""산출물만으로 채점하는 경로 (docs/dev/AI-1)."""

import pytest

from family_fitness_ai.stats.assess import Cell, Reference, assess, score_one

REF = Reference("data/release")


def score_of(cell: Cell, value: float) -> float:
    return score_one(cell, value).score


def test_연령대는_산출물의_구간에서_되찾는다() -> None:
    assert REF.age_group_of(11, "세") == "유소년"
    assert REF.age_group_of(41, "세") == "성인"
    assert REF.age_group_of(60, "개월") == "유아기"
    assert REF.age_group_of(8, "세") is None  # 만 7~10세는 측정 0건
    assert REF.age_group_of(70, "세") is None  # 어르신은 범위 밖


def test_구간은_기준표의_밴드를_따른다() -> None:
    assert REF.band_of("유소년", 11) == (11, 11)
    assert REF.band_of("성인", 41) == (40, 44)
    assert REF.band_of("유아기", 60) == (60, 65)


def test_문턱값은_정확히_문턱_점수를_받는다() -> None:
    """산출물 경로로도 40·60·80 이 그대로 나와야 한다."""
    cell = REF.cell("유소년", 11, "F", "028")
    assert cell is not None
    assert [score_of(cell, v) for v in (34.8, 39.5, 44.4)] == [40.0, 60.0, 80.0]


def test_3등급이_없는_항목은_앵커가_넷이다() -> None:
    운동체력 = REF.cell("유소년", 11, "F", "022")
    건강체력 = REF.cell("유소년", 11, "F", "028")
    assert 운동체력 is not None and 건강체력 is not None
    assert 운동체력.anchors.y == (0.0, 60.0, 80.0, 100.0)
    assert 건강체력.anchors.y == (0.0, 40.0, 60.0, 80.0, 100.0)


def test_한_사람을_채점한다() -> None:
    got = assess(
        REF,
        age=11,
        age_unit="세",
        sex="F",
        measurements={"028": 52.3, "012": 12.6, "020": 66, "018": 15.6},
    )
    assert got.age_group == "유소년"
    assert got.input_level == "L2"
    assert [f.item_code for f in got.factors] == ["028", "012", "020"]
    assert got.not_scored == ["018"]  # 신체조성은 점수화하지 않는다
    assert got.factors[0].score > got.factors[-1].score  # 점수 내림차순
    assert got.focus_one == got.factors[-1].factor


def test_점수를_낼_수_없는_나이는_사유를_준다() -> None:
    got = assess(REF, age=8, age_unit="세", sex="F", measurements={"028": 40.0})
    assert got.age_group is None and got.note
    assert got.factors == []


def test_산출물이_없으면_바로_멈춘다(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="산출물이 없다"):
        Reference(tmp_path)
