"""영상 토막을 가르는 규칙 (docs/05).

**만들어 넣은 작은 표로만 시험한다.** 낸 파일(`data/release/video_segments.csv`)을
읽는 시험은 데이터를 다시 처리한 뒤에 쓴다 — 지금은 없다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from family_fitness_ai.mission import segments as S

RELEASE = Path(__file__).resolve().parents[1] / "data" / "release"
SEGMENTS = RELEASE / S.SEGMENTS_FILE

# 실측 (`python -m family_fitness_ai.mission.segments`):
# 운동×영상 142행 중 시각이 붙은 127행 → 같은 (영상, 시작) 5쌍을 합쳐 122구간.
SEGMENT_COUNT = 122
LAST_COUNT = 30  # 영상 30편


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(SEGMENTS, encoding="utf-8-sig")


def test_파일이_비어_있어도_빈_목록을_돌려준다(tmp_path: Path) -> None:
    """미션은 영상이 없어도 성립한다 — 읽기가 예외를 던지면 그것까지 못 만든다."""
    empty = tmp_path / S.SEGMENTS_FILE
    empty.write_text("", encoding="utf-8")
    assert S.read_segments(empty) == []
    assert S.read_segments(tmp_path / "없는파일.csv") == []


# ── 규칙 ────────────────────────────────────────────────────────────


def exercises(*rows: dict[str, object]) -> pd.DataFrame:
    blank = {
        "exercise_name": "",
        "video_id": "v1",
        "source": "screen",
        "common": "False",
        "evidence_text": "",
        "fitness_factors": "",
        "age_group": "유아기",
        "start_sec": "",
        "end_sec": "",
        "url": "",
    }
    return pd.DataFrame([{**blank, **r} for r in rows])


def build(*rows: dict[str, object], length: int = 600) -> list[S.VideoSegment]:
    labeling = pd.DataFrame([{"video_id": "v1", "duration_sec": str(length)}])
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "video:v1",
                "source": "video",
                "citation_label": "국민체력100 운동영상 · 운동",
            }
        ]
    )
    got, _ = S.build(exercises(*rows), labeling, chunks)
    return got


def test_끝은_다음_운동의_시작이고_마지막은_이름표_끝이다() -> None:
    """이름표가 사라진 시점으로 재면 운동이 이어지는데도 짧게 나온다 (§5.2)."""
    got = build(
        {"exercise_name": "가", "start_sec": "10", "end_sec": "40"},
        {"exercise_name": "나", "start_sec": "100", "end_sec": "150"},
    )
    assert [(s.start_sec, s.end_sec, s.duration_sec, s.gap_sec, s.length_basis) for s in got] == [
        (10, 100, 90, 60, "next_start"),  # 이름표 끝(40)이 아니라 다음 시작(100)으로 잰다
        (100, 150, 50, 0, "label_end"),
    ]
    assert [s.last for s in got] == [False, True]


def test_영상_길이를_넘으면_영상_끝으로_닫는다() -> None:
    """이름표 끝이 영상 길이보다 뒤에 적혀 있을 때. 잰 값이 서로 어긋나면 근거를 남긴다."""
    (segment,) = build({"exercise_name": "가", "start_sec": "10", "end_sec": "120"}, length=100)
    assert (segment.end_sec, segment.length_basis, segment.duration_sec) == (100, "over_cap", 90)


def test_시각이_없는_행과_청크가_없는_영상은_세어서_뺀다() -> None:
    labeling = pd.DataFrame([{"video_id": "v1", "duration_sec": "600"}])
    chunks = pd.DataFrame([{"chunk_id": "video:v1", "source": "video", "citation_label": "ㄱ"}])
    got, dropped = S.build(
        exercises(
            {"exercise_name": "가", "start_sec": "10", "end_sec": "40"},
            {"exercise_name": "나"},  # 시각이 없다
            {"exercise_name": "다", "video_id": "v2", "start_sec": "10", "end_sec": "40"},
        ),
        labeling,
        chunks,
    )
    assert [s.exercise_name for s in got] == ["가"]
    assert dropped == {"시각 없음": 1, "청크 없음": 1, "이름이 둘": 0}


def test_이름이_둘이면_비공통_어휘_완전일치_근거_순서로_고른다() -> None:
    """실측에서 걸리는 5쌍은 `척추 들어올리기 (고양이자세)` 꼴이다 (docs/02)."""
    pair = [
        {
            "exercise_name": "고양이 자세",
            "common": True,
            "evidence_text": "03:08 척추 들어올리기 (고양이자세)",
        },
        {
            "exercise_name": "척추 들어올리기",
            "common": True,
            "evidence_text": "03:08 척추 들어올리기 (고양이자세)",
        },
    ]
    # 둘 다 공통이고 둘 다 통째로는 어휘와 다르니 근거에 먼저 적힌 이름이 남는다
    assert S.pick_one(pair)["exercise_name"] == "척추 들어올리기"

    # 비공통이 먼저다
    assert S.pick_one([{**pair[0], "common": False}, pair[1]])["exercise_name"] == "고양이 자세"

    # 통째로 어휘와 같은 이름이 그다음이다
    whole = {"exercise_name": "고양이자세", "common": True, "evidence_text": "03:08 고양이 자세"}
    assert S.pick_one([pair[1], whole])["exercise_name"] == "고양이자세"


def test_같은_시작에_이름이_둘이면_한_행만_낸다() -> None:
    got = build(
        {
            "exercise_name": "고양이 자세",
            "common": "True",
            "evidence_text": "03:08 척추 들어올리기 (고양이자세)",
            "start_sec": "188",
            "end_sec": "238",
        },
        {
            "exercise_name": "척추 들어올리기",
            "common": "True",
            "evidence_text": "03:08 척추 들어올리기 (고양이자세)",
            "start_sec": "188",
            "end_sec": "238",
        },
    )
    assert [(s.start_sec, s.exercise_name) for s in got] == [(188, "척추 들어올리기")]


# ── 통계 ────────────────────────────────────────────────────────────
