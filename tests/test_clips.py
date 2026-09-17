"""화면 글자 → 클립.

시험 자료는 지어낸 것이 아니라 실제 OCR 결과를 잘라 둔 것이다
(tests/fixtures/screen_frames.json). 화면이 어떻게 생겼는지를 두고 다투지 않으려면
진짜 화면을 가져다 두는 편이 낫다. 세 가지 모양을 담았다.

    program    제목 칸에는 프로그램 이름과 단계만 뜨고, 운동 이름은 자막 칸에 뜬다
    single     운동이 하나뿐이라 그 이름이 영상 내내 떠 있다
    preschool  운동 이름이 제목 칸에 뜬다

여기서 가르는 것은 **얼마나 오래 떠 있느냐**뿐이다. 오래 떠 있지만 운동이 아닌
글자는 여기서 걸러지지 않는다 — 그건 다음 층인 clip_labels 의 is_exercise 가
맡는다. 한 층이 다 하려 들면 둘 다 나빠진다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from family_fitness_ai.video.clips import clips_of_video

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "screen_frames.json").read_text(encoding="utf-8")
)


def clips(key: str):
    screen = FIXTURE[key]
    return clips_of_video(screen["video_id"], screen)


def test_program_video_reads_names_from_the_caption_bar():
    """제목 칸에는 「비대면 표준 운동 프로그램 / 준비운동」만 뜬다."""
    found = clips("program")
    assert [clip.name for clip in found][:3] == [
        "옆으로 누워 발 뒤로 넘기기",
        "넙다리 안쪽 늘리기 (나비자세)",
        "척추 들어올리기 (고양이자세)",
    ]
    assert {clip.phase for clip in found} == {"준비운동"}


def test_clips_do_not_overlap_and_run_forward():
    found = clips("program")
    for before, after in zip(found, found[1:], strict=False):
        assert before.end_sec <= after.start_sec


def test_single_exercise_video_keeps_its_only_name():
    """이름 하나가 영상의 92%를 차지한다. 배너가 아니라 그게 운동이다."""
    found = clips("single")
    assert [clip.name for clip in found] == ["거북이 스트레칭"]
    assert found[0].duration_sec > 60


def test_preschool_video_reads_names_from_the_title_box():
    assert [clip.name for clip in clips("preschool")] == [
        "인사 체조",
        "흔들어 체조",
        "거북이 스트레칭",
    ]


@pytest.mark.parametrize("key", list(FIXTURE))
def test_every_clip_is_long_enough_to_be_worth_showing(key: str):
    assert all(clip.duration_sec >= 20 for clip in clips(key))


def test_narration_that_never_settles_is_not_a_name():
    noisy = {
        "interval_sec": 2,
        "frames": [{"t": t, "texts": [], "bar": [f"해설 {t}"]} for t in range(0, 60, 2)],
    }
    assert clips_of_video("vid", noisy) == []
