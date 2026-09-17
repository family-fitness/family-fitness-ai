from conftest import needs_release

from family_fitness_ai.stats import tables

pytestmark = needs_release


def test_percentile_matches_the_contract_example():
    """명세 §1 예시 — 유소년 11세 여아, 왕복오래달리기 70회."""
    sample = tables.peer("유소년", "F", 11, "020")
    assert sample is not None
    assert sample.n > 1000
    assert tables.percentile_of(sample, 70.0, lower_is_better=False) == 80


def test_lower_is_better_flips_the_percentile():
    sample = tables.peer("청소년", "M", 15, "017")
    if sample is None:
        return
    fast = tables.percentile_of(sample, sample.quantiles[10], lower_is_better=True)
    slow = tables.percentile_of(sample, sample.quantiles[90], lower_is_better=True)
    assert fast > slow


def test_grade_needs_the_whole_battery():
    full = {"020": 70, "028": 45, "009": 40, "012": 12, "043": 33, "022": 170, "044": 20}
    assert tables.certify("유소년", "F", 11, full) == "1등급"
    # 집에서 두어 개만 잰 사람은 맨 아래로 내리지 않는다 — 판정하지 않는다.
    assert tables.certify("유소년", "F", 11, {"020": 70, "012": 12}) is None


def test_grade_falls_to_participation_when_nothing_is_met():
    weak = {"020": 1, "028": 1, "009": 0, "012": -10, "043": 1, "022": 10, "044": 0}
    assert tables.certify("유소년", "F", 11, weak) == "참가"


def test_third_grade_looks_at_body_composition_instead_of_skill_items():
    """3등급 줄에는 민첩성·순발력·협응력 기준이 없고 BMI·체지방률이 대신 있다."""
    rows = tables.grade_thresholds("성인", "M", "3등급", 30)
    codes = {row.item_code for row in rows}
    assert "018" in codes
    assert not {"021", "040", "022", "041"} & codes
