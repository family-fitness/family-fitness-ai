from conftest import needs_release

from family_fitness_ai.stats.assess import Profile, assessment, trajectory

pytestmark = needs_release

CHILD = Profile(
    profile_ref="p_c7a91f",
    age=11,
    age_unit="세",
    sex="F",
    height_cm=148.0,
    weight_kg=41.0,
    measurements={"028": 41.3, "012": 4.0, "020": 70, "022": 133, "009": 30},
)


def test_measured_profile_gets_factor_rows_and_a_focus():
    result = assessment(CHILD)
    assert result["input_level"] == "L2"
    assert result["age_group"] == "유소년"
    factors = result["parent_scope"]["factors"]
    assert {row["factor"] for row in factors} >= {"유연성", "심폐지구력"}
    assert result["child_scope"]["focus_one"]["factor"] == "유연성"
    assert "지금 키우기 좋은 영역" in result["parent_scope"]["copy"]["focus"]


def test_no_measurement_means_no_factor_is_named():
    """근거 없이 고른 요인에는 인용할 것이 없다."""
    result = assessment(Profile("p", 8, "세", "M"))
    assert result["input_level"] == "L0"
    assert result["child_scope"]["focus_one"] is None
    assert result["parent_scope"]["factors"] == []
    assert result["parent_scope"]["grade"] is None


def test_copy_never_uses_the_banned_words():
    from family_fitness_ai.common.copy import banned_words_in

    result = assessment(CHILD)
    for text in result["parent_scope"]["copy"].values():
        assert banned_words_in(text) == []
    assert banned_words_in(result["child_scope"]["focus_one"]["copy"]) == []


def test_trajectory_is_a_group_distribution_not_a_forecast():
    result = trajectory(CHILD, "028", 3)
    assert result["basis"] == "cross_sectional_group_distribution"
    assert "개인의 변화를 나타내지 않습니다" in result["notice"]
    ages = [band["age"] for band in result["bands"]]
    assert ages == sorted(ages)
