from family_fitness_ai.common.items import (
    AGE_GROUP_ITEMS,
    ITEMS,
    age_group_of,
    item_label,
    scored_items,
)


def test_age_group_boundaries():
    assert age_group_of(60, "개월") == "유아기"
    assert age_group_of(6, "세") == "유아기"
    assert age_group_of(7, "세") == "유소년"
    assert age_group_of(12, "세") == "유소년"
    assert age_group_of(13, "세") == "청소년"
    assert age_group_of(19, "세") == "성인"
    assert age_group_of(65, "세") == "어르신"


def test_shuttle_run_label_changes_by_age_group():
    assert item_label("020", "유아기") == "10m왕복오래달리기"
    assert item_label("020", "유소년") == "15m왕복오래달리기"
    assert item_label("020", "청소년") == "20m왕복오래달리기"


def test_coordination_item_differs_by_age_group():
    """같은 이름의 시험이 유소년은 회, 청소년은 초다."""
    assert ITEMS["044"].unit == "회" and not ITEMS["044"].lower_is_better
    assert ITEMS["017"].unit == "초" and ITEMS["017"].lower_is_better
    assert "044" in AGE_GROUP_ITEMS["유소년"]
    assert "017" in AGE_GROUP_ITEMS["청소년"]


def test_scored_items_keeps_only_the_age_groups_battery():
    values = {"020": 70, "012": 4, "041": 0.5, "018": 17.0}
    assert scored_items("유소년", values) == ["020", "012"]
