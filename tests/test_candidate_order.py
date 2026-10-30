"""코치에게 넘기는 후보의 차례(catalog._lead).

측정으로 고른 「지금 키우기 좋은 영역」이면 본운동 후보의 앞쪽 어디를 잘라도 그 요인이
절반 이상이 되게 세운다. 이미 앞서 있는 것은 깎지 않는다. 순위대로만 두면 처방 동작이
앞을 차지해 앞 20편 가운데 그 요인이 28%였고, 코치가 고른 뒤 채우기가 편성당 3편 넘게
바꿔 넣었다(src/script/evaluate_coach.py).
"""

from __future__ import annotations

import pytest
from conftest import needs_index, needs_release

from family_fitness_ai.video import catalog


def _clip(video_id: str, factor: str, phase: str = "본운동") -> catalog.Clip:
    return catalog.Clip(
        video_id=video_id,
        name=f"{factor} {video_id}",
        exercise_name="",
        fitness_factor=factor,
        phase=phase,
        start_sec=0,
        end_sec=40,
        age_group="유소년",
        quiet=True,
        home_ok=True,
        needs_props=False,
    )


#: 순위대로 선 본운동 후보. 처방 동작(근력)이 앞을 차지했다.
RANKED = [
    _clip("a", "근력"),
    _clip("b", "근력"),
    _clip("c", "근력"),
    _clip("x", "유연성"),
    _clip("y", "유연성"),
    _clip("d", "근력"),
]


def _ids(clips: list[catalog.Clip]) -> list[str]:
    return [clip.video_id for clip in clips]


def test_the_factor_is_pulled_forward_until_it_is_half():
    assert _ids(catalog._lead(RANKED, "유연성", ())) == ["x", "a", "y", "b", "c", "d"]


def test_clips_already_in_front_keep_their_place():
    # 그 요인이 이미 앞서 있으면 순위대로 둔다. 절반으로 깎지 않는다.
    front = [_clip("x", "유연성"), _clip("y", "유연성"), _clip("z", "유연성"), _clip("a", "근력")]
    assert _ids(catalog._lead(front, "유연성", ())) == ["x", "y", "z", "a"]


def test_within_each_side_the_rank_order_holds():
    woven = catalog._lead(RANKED, "유연성", ())
    assert _ids([c for c in woven if c.fitness_factor == "유연성"]) == ["x", "y"]
    assert _ids([c for c in woven if c.fitness_factor == "근력"]) == ["a", "b", "c", "d"]


def test_recent_videos_stay_at_the_back_whatever_their_factor():
    woven = catalog._lead(RANKED, "유연성", ("x",))
    assert _ids(woven) == ["y", "a", "b", "c", "d", "x"]


def test_other_phases_keep_their_places():
    warm = _clip("w", "유연성", "준비운동")
    cool = _clip("z", "유연성", "정리운동")
    woven = catalog._lead([warm, *RANKED, cool], "유연성", ())
    assert woven[0] is warm and woven[-1] is cool
    assert _ids(woven[1:-1]) == ["x", "a", "y", "b", "c", "d"]


def test_without_a_measured_factor_nothing_moves():
    assert catalog._lead(RANKED, "", ()) is RANKED


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
def test_any_front_slice_is_at_least_half_the_factor_while_it_lasts(cut: int):
    many = [_clip(f"s{n}", "근력") for n in range(6)] + [_clip(f"f{n}", "유연성") for n in range(6)]
    front = catalog._lead(many, "유연성", ())[:cut]
    assert sum(c.fitness_factor == "유연성" for c in front) >= catalog.grow_quota(cut)


def _prescribed() -> set[str]:
    """유소년 11세 남자의 처방 근거 동작. 편성은 이 이름의 클립을 앞세운다."""
    from family_fitness_ai.rag.index import corpus

    chunks = [c for c in corpus().chunks if c.chunk_id.startswith("prescription:유소년-11-M-")]
    return catalog.prescribed_names(chunks[:4])


def _front(factor: str, grow: str) -> list[catalog.Clip]:
    clips, _ = catalog.pool("유소년", factor=factor, prescribed=_prescribed(), grow=grow)
    return [c for c in clips if c.phase == "본운동"][:20]


@needs_release
@needs_index
def test_a_factor_the_prescription_pushed_back_comes_to_the_front():
    # 처방 동작이 앞을 차지해 민첩성 클립이 앞 20편에 하나도 없었다.
    assert sum(c.fitness_factor == "민첩성" for c in _front("민첩성", "")) == 0
    assert sum(c.fitness_factor == "민첩성" for c in _front("민첩성", "민첩성")) >= 9


@needs_release
@needs_index
def test_a_factor_already_in_front_is_not_cut_to_half():
    # 근력은 처방 동작에도 많아 앞 20편 가운데 13편이었다. 절반(10)으로 깎지 않는다.
    before = sum(c.fitness_factor == "근력" for c in _front("근력", ""))
    after = sum(c.fitness_factor == "근력" for c in _front("근력", "근력"))
    assert after >= before >= 10


@needs_release
@needs_index
def test_only_the_order_changes():
    ranked, _ = catalog.pool("유소년", factor="민첩성", prescribed=_prescribed())
    woven, _ = catalog.pool("유소년", factor="민첩성", prescribed=_prescribed(), grow="민첩성")
    assert sorted(_ids(woven)) == sorted(_ids(ranked))
