"""`exercise_videos` 적재 형식 (docs/05 · ReBuild §1.2 ④)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from family_fitness_ai.common.settings import RELEASE_DIR
from family_fitness_ai.mission.loadout import (
    AGE_RANGE,
    SQL_FILE,
    VIDEO_COLUMNS,
    VIDEOS_FILE,
    build,
    to_sql,
)
from family_fitness_ai.mission.segments import VideoSegment


def _segment(video_id: str, age_group: str, start: int = 0) -> VideoSegment:
    return VideoSegment(
        video_id=video_id,
        start_sec=start,
        end_sec=start + 60,
        label_end_sec=start + 60,
        length_basis="next_start",
        last=False,
        exercise_name="운동",
        age_group=age_group,
        source="screen",
        common=False,
        evidence_text="",
        chunk_id=f"video:{video_id}",
        citation_label=f"국민체력100 운동영상 · {video_id}",
    )


def _labeling(*video_ids: str, factors: str = "근력;유연성") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": v,
                "title": f"제목 {v}",
                "duration_sec": 600,
                "fitness_factors": factors,
                "labeler_version": "labeler:rules/v3",
            }
            for v in video_ids
        ]
    )


def test_영상_한_편이_한_행이다() -> None:
    segments = [_segment("v1", "유아기", 0), _segment("v1", "유아기", 60)]
    frame = build(segments, _labeling("v1"))
    assert len(frame) == 1
    assert list(frame.columns) == VIDEO_COLUMNS


def test_연령_폭이_백엔드_AgeRange_와_같다() -> None:
    """두 벌이 되면 우리가 낸 라벨이 백엔드 필터를 통과하지 못한다."""
    for group, (low, high) in AGE_RANGE.items():
        frame = build([_segment("v", group)], _labeling("v"))
        assert (int(frame.age_from.iloc[0]), int(frame.age_to.iloc[0])) == (low, high)


def test_구간이_여러_연령대면_연령을_비운다() -> None:
    """한 영상의 연령을 하나로 말할 수 없으면 비운다 — 비면 아이에게 나가지 않는다."""
    segments = [_segment("v", "유아기", 0), _segment("v", "청소년", 60)]
    frame = build(segments, _labeling("v"))
    assert pd.isna(frame.age_from.iloc[0])
    assert pd.isna(frame.age_to.iloc[0])


def test_잰_값이_없는_열은_비어_있다() -> None:
    """`intensity`·`space`·`noise`·`equipment` 를 더미로 채우지 않는다."""
    frame = build([_segment("v", "유아기")], _labeling("v"))
    for column in ("intensity", "space", "noise", "equipment"):
        assert frame[column].isna().all(), f"{column} 에 값이 들어갔다"


def test_요인은_쉼표로_잇는다() -> None:
    """`factors` 는 varchar(120) 에 쉼표로 이은 한글 요인이다 — 우리 산출은 세미콜론."""
    frame = build([_segment("v", "유아기")], _labeling("v", factors="근력;유연성"))
    assert frame.factors.iloc[0] == "근력,유연성"


def test_빈_칸이_nan_문자열이_되지_않는다() -> None:
    """pandas 의 `NaN` 은 truthy 다 — `str(v or "")` 로는 문자열 `nan` 이 들어간다."""
    labeling = _labeling("v")
    labeling.loc[:, "fitness_factors"] = None
    labeling.loc[:, "title"] = None
    frame = build([_segment("v", "유아기")], labeling)
    assert frame.factors.iloc[0] == ""
    assert frame.title.iloc[0] == ""
    assert "nan" not in frame.astype(str).to_numpy()


def test_라벨링에_없는_영상도_행이_난다() -> None:
    frame = build([_segment("모르는영상", "유아기")], _labeling("다른영상"))
    assert len(frame) == 1
    assert frame.title.iloc[0] == ""


def test_sql_은_멱등이다() -> None:
    """두 번 돌려도 행 수가 같아야 한다 — `ON CONFLICT` 로 갱신한다."""
    frame = build([_segment("v1", "유아기"), _segment("v2", "청소년")], _labeling("v1", "v2"))
    sql = to_sql(frame)
    assert "ON CONFLICT (video_id) DO UPDATE SET" in sql
    assert sql.count("INSERT INTO exercise_videos") == 1
    assert sql.rstrip().endswith(";")
    for column in VIDEO_COLUMNS:
        if column != "video_id":
            assert f"{column} = EXCLUDED.{column}" in sql


def test_sql_이_따옴표를_탈출한다() -> None:
    labeling = _labeling("v")
    labeling.loc[:, "title"] = "아이's 운동"
    sql = to_sql(build([_segment("v", "유아기")], labeling))
    assert "'아이''s 운동'" in sql


def test_sql_은_빈_값을_NULL_로_쓴다() -> None:
    sql = to_sql(build([_segment("v", "유아기")], _labeling("v")))
    assert "NULL" in sql
    assert "'nan'" not in sql
    assert "'None'" not in sql


@pytest.mark.skipif(not (RELEASE_DIR / VIDEOS_FILE).exists(), reason="적재 산출물이 아직 없다")
def test_낸_산출물이_계약_열을_그대로_쓴다() -> None:
    frame = pd.read_csv(RELEASE_DIR / VIDEOS_FILE, encoding="utf-8-sig")
    assert list(frame.columns) == VIDEO_COLUMNS
    assert len(frame) > 0
    assert not frame.video_id.duplicated().any()
    # 연령 폭이 있는 행은 전부 AgeRange 의 값 중 하나여야 한다
    known = set(AGE_RANGE.values())
    aged = frame[frame.age_from.notna()]
    assert {(int(a), int(b)) for a, b in zip(aged.age_from, aged.age_to, strict=True)} <= known
    assert (RELEASE_DIR / SQL_FILE).exists()


@pytest.mark.skipif(not (RELEASE_DIR / SQL_FILE).exists(), reason="적재 SQL 이 아직 없다")
def test_낸_sql_이_한_문장이다() -> None:
    sql = Path(RELEASE_DIR / SQL_FILE).read_text(encoding="utf-8")
    assert sql.count(";") == 1
    assert "ON CONFLICT" in sql
