"""clip_labels.csv 의 빈 요인을 공단 표의 같은 동작 요인으로 채운다.

유튜브 클립 라벨은 이름을 처방 어휘에 이을 뿐(exact, embed) 요인을 붙이지 않았다.
그래서 「다리뻗어 상체 숙이기」 같은 줄은 요인 없이 나가 요인으로 고를 때 후보가
되지 못했다. 공단 영상은 한 편이 한 동작이고 API 가 요인을 준다. 같은 이름의 공단
클립에 요인이 하나로 정해진 동작만 그 요인을 빌려 온다. 둘로 갈리면 어느 쪽인지
알 수 없으니 비워 둔다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from conftest import needs_release

from family_fitness_ai.common.settings import settings
from family_fitness_ai.video import labels

KNOWN = {
    "걷기": frozenset({"심폐지구력"}),
    "엎드려 상체 들어올리기": frozenset({"근력", "근지구력", "유연성"}),
}


def test_an_empty_factor_takes_the_only_kspo_factor():
    assert labels.factor_from_kspo("걷기", "", KNOWN) == "심폐지구력"


def test_a_move_split_between_factors_stays_empty():
    assert labels.factor_from_kspo("엎드려 상체 들어올리기", "", KNOWN) == ""


def test_a_move_kspo_does_not_have_stays_empty():
    assert labels.factor_from_kspo("어깨 스트레칭", "", KNOWN) == ""


def test_a_factor_already_set_is_kept():
    assert labels.factor_from_kspo("걷기", "근력", KNOWN) == "근력"


def test_a_label_without_an_exercise_name_stays_empty():
    assert labels.factor_from_kspo("", "", KNOWN) == ""


KSPO_HEADER = "video_id,exercise_name,fitness_factor,is_exercise\n"


def test_kspo_factors_skip_blank_factors_and_non_exercise_rows(tmp_path: Path):
    path = tmp_path / "kspo_videos.csv"
    path.write_text(
        KSPO_HEADER
        + "a,걷기,심폐지구력,True\n"
        + "b,걷기,,True\n"
        + "c,걷기,근력,False\n"
        + "d,무릎 높여 제자리 달리기,민첩성,True\n"
        + "d,무릎 높여 제자리 달리기,순발력,True\n",
        encoding="utf-8",
    )
    assert labels.kspo_factors(path) == {
        "걷기": frozenset({"심폐지구력"}),
        "무릎 높여 제자리 달리기": frozenset({"민첩성", "순발력"}),
    }


LABEL_ROWS = (
    "name_on_video,exercise_name,fitness_factor,phase,is_exercise,home_ok,quiet,needs_props,source,score\n"
    "걷기를 해요,걷기,,본운동,True,True,True,False,embed,0.9111\n"
    "엎드려 상체 들어올리기,엎드려 상체 들어올리기,,본운동,True,True,True,False,exact,1.0\n"
    "휴식,,,본운동,False,True,True,False,llm,0.0\n"
)


def test_fill_factors_changes_only_the_empty_factor_cell(tmp_path: Path):
    (tmp_path / "clip_labels.csv").write_text(LABEL_ROWS, encoding="utf-8", newline="")
    (tmp_path / "kspo_videos.csv").write_text(
        KSPO_HEADER
        + "a,걷기,심폐지구력,True\n"
        + "b,엎드려 상체 들어올리기,근력,True\n"
        + "c,엎드려 상체 들어올리기,유연성,True\n",
        encoding="utf-8",
    )
    assert labels.fill_factors(tmp_path) == ["걷기를 해요"]
    after = (tmp_path / "clip_labels.csv").read_text(encoding="utf-8")
    assert after == LABEL_ROWS.replace("걷기,,본운동", "걷기,심폐지구력,본운동")
    # 두 번 돌려도 같다
    assert labels.fill_factors(tmp_path) == []


# 실제 표. BE V169 가 DB 에만 채운 동작과 견준다.


def _release_rows() -> list[dict[str, str]]:
    with (settings().release_dir / "clip_labels.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@needs_release
@pytest.mark.parametrize(
    ("exercise_name", "factor"),
    [
        ("다리뻗어 상체 숙이기", "유연성"),
        ("다리 벌려 앞으로 상체 숙이기", "유연성"),
        ("나비자세", "유연성"),
        ("걷기", "심폐지구력"),
        ("의자 앞에서 앉았다 일어서기", "근력"),
    ],
)
def test_the_table_fills_moves_with_one_kspo_factor(exercise_name: str, factor: str):
    rows = [row for row in _release_rows() if row["exercise_name"] == exercise_name]
    assert rows
    assert {row["fitness_factor"] for row in rows} == {factor}


@needs_release
@pytest.mark.parametrize(
    "exercise_name",
    [
        # 공단 표에서 요인이 둘 이상이다. BE V169 는 뒤의 둘을 민첩성과 근력으로 채웠지만
        # 공단 표에는 순발력과 근지구력도 함께 붙어 있다.
        "엎드려 상체 들어올리기",
        "무릎 높여 제자리 달리기",
        "윗몸 말아 올리기",
        # 공단 영상에 없는 동작
        "어깨 스트레칭",
        "거북이 스트레칭",
    ],
)
def test_the_table_leaves_moves_without_one_kspo_factor_empty(exercise_name: str):
    rows = [row for row in _release_rows() if row["exercise_name"] == exercise_name]
    assert rows
    assert any(not row["fitness_factor"] for row in rows)


@needs_release
def test_no_label_is_left_empty_when_kspo_has_one_factor():
    known = labels.kspo_factors(settings().release_dir / "kspo_videos.csv")
    left = [
        row["name_on_video"]
        for row in _release_rows()
        if not row["fitness_factor"] and len(known.get(row["exercise_name"], ())) == 1
    ]
    assert left == []
