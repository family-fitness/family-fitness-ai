"""영상 구간 — 끝 시각과 길이 근거 (docs/05 §3.2).

낸 파일(`data/release/video_segments.csv`)을 그대로 읽어 단정한다 — 백엔드와 검수가
읽는 것이 그 파일이다. 규칙 자체는 만들어 넣은 작은 표로 따로 시험한다.
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


# ── 낸 파일 ──────────────────────────────────────────────────────────


def test_모든_구간에_끝_시각이_있다(frame: pd.DataFrame) -> None:
    """[AI-7] 은 시작만 냈다. 끝이 없으면 몇 분 하는 운동인지 모른다."""
    assert len(frame) == SEGMENT_COUNT
    assert frame["end_sec"].notna().sum() == SEGMENT_COUNT
    assert frame["label_end_sec"].notna().sum() == SEGMENT_COUNT
    assert (frame["end_sec"] > frame["start_sec"]).all()


def test_길이_근거는_셋_중_하나다(frame: pd.DataFrame) -> None:
    assert set(frame["length_basis"]) <= set(S.LENGTH_BASES)
    counts = frame["length_basis"].value_counts().to_dict()
    # 마지막 구간만 이름표 끝으로 닫힌다. 영상 길이를 넘어가는 구간은 없다
    assert counts == {"next_start": SEGMENT_COUNT - LAST_COUNT, "label_end": LAST_COUNT}


def test_같은_영상_같은_시작이_두_행이_되지_않는다(frame: pd.DataFrame) -> None:
    assert not frame.duplicated(["video_id", "start_sec"]).any()


def test_마지막_구간의_gap_은_전부_0_이다(frame: pd.DataFrame) -> None:
    """**결함을 시험으로 고정해 둔다** (docs/05 §3.2). 마지막 구간은 이름표 끝으로
    닫으므로 gap 이 언제나 0 이고, 그래서 gap 으로 거르는 검사가 마지막 구간을 하나도
    걸러내지 못한다. 고친 것이 아니라 드러낸 것이다.
    """
    last = frame[frame["last"]]
    assert len(last) == LAST_COUNT
    assert (last["gap_sec"] == 0).all()
    assert (last["end_sec"] == last["label_end_sec"]).all()
    # gap 이 0 인 구간이 정확히 그 마지막 구간들이다
    assert (frame["gap_sec"] == 0).sum() == LAST_COUNT


def test_길이는_끝에서_시작을_뺀_값이다(frame: pd.DataFrame) -> None:
    assert (frame["duration_sec"] == frame["end_sec"] - frame["start_sec"]).all()
    assert (frame["gap_sec"] == frame["end_sec"] - frame["label_end_sec"]).all()


def test_재지_않은_칸은_열로도_만들지_않는다(frame: pd.DataFrame) -> None:
    """`space`·`noise`·`equipment`·`intensity` 는 규칙도 잰 값도 없다 (AGENTS.md §4)."""
    assert list(frame.columns) == S.SEGMENT_COLUMNS
    assert not {"space", "noise", "equipment", "intensity"} & set(frame.columns)


def test_주소는_그_구간부터_재생된다(frame: pd.DataFrame) -> None:
    """키는 `video_id`·`start_sec` 이고 주소는 딸린 값이다 (docs/03 §4.2)."""
    for row in frame.to_dict("records"):
        assert row["url"] == (
            f"https://www.youtube.com/watch?v={row['video_id']}&t={row['start_sec']}s"
        )


def test_인용_이름표와_청크가_붙어_있다(frame: pd.DataFrame) -> None:
    """인용할 청크가 없는 구간은 미션에 쓸 수 없다 (docs/04 §2.2)."""
    assert (frame["chunk_id"] == "video:" + frame["video_id"]).all()
    assert frame["citation_label"].str.len().min() > 0
    assert frame["age_group"].notna().all()


def test_파일이_비어_있어도_빈_목록을_돌려준다(tmp_path: Path) -> None:
    """미션은 영상이 없어도 성립한다 — 읽기가 예외를 던지면 그것까지 못 만든다."""
    empty = tmp_path / S.SEGMENTS_FILE
    empty.write_text("", encoding="utf-8")
    assert S.read_segments(empty) == []
    assert S.read_segments(tmp_path / "없는파일.csv") == []


def test_낸_파일을_다시_읽으면_같은_구간이다(frame: pd.DataFrame) -> None:
    got = S.read_segments(SEGMENTS)
    assert len(got) == SEGMENT_COUNT
    assert S.segments_frame(got).equals(frame)


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


def test_통계는_문서가_주장한_값을_나란히_싣는다() -> None:
    """수치를 맞추려고 코드를 비틀지 않는다 — 어긋난 곳이 보이게만 한다."""
    rows = S.stats(S.read_segments(SEGMENTS))
    got = {(g, m): v for g, m, v in rows}
    assert got[("전체", "구간 수")] == "122"
    assert got[("전체", "gap 중앙")] == "6"
    assert got[("전체", "gap ≤ 60 통과")] == "94"
    assert got[("마지막 구간", "gap 이 0")] == str(LAST_COUNT)
    assert got[("마지막 구간", "gap ≤ 60 통과")] == str(LAST_COUNT)  # 하나도 걸러지지 않는다
    assert got[("마지막 구간", "gap 0 인 구간 중 마지막이 아닌 것")] == "0"
    frame = S.stats_frame(rows)
    claimed = frame[frame["claimed_in_AI-14"] != ""]
    assert len(claimed) == len(S.CLAIMED)
