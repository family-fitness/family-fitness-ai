"""공통 준비·마무리 표시 (docs/dev/AI-7 §3.10).

긴 에피소드들은 같은 여는 노래·마무리 체조를 모든 영상에 넣는다. 틀린 이름은 아니지만 그
영상의 본운동이 아니라 뒤로 미룬다.
"""

from __future__ import annotations

import pandas as pd

from family_fitness_ai.labeling import collect as C
from family_fitness_ai.labeling import label as L

PLAYLIST = ("PLBdpvOnWjVZmeshOCp06HAov4ieoNHoen", "🎯 연령별 맞춤 운동 : 영유아")
VOCAB = L.Vocabulary(["거북이 스트레칭", "엉금엉금 기어요"])


def build(specs: dict[str, tuple[str, list[str]]]) -> list[L.VideoLabel]:
    """영상 id → (제목, 화면에 차례로 뜨는 이름표). 이름표마다 두 프레임씩 읽힌다."""
    rows, screens = [], {}
    for video_id, (title, names) in specs.items():
        rows.append(
            {
                **dict.fromkeys(C.VIDEO_COLUMNS, ""),
                "video_id": video_id,
                "title": title,
                "playlist_ids": PLAYLIST[0],
                "playlist_titles": PLAYLIST[1],
            }
        )
        frames = [
            {"t": i * 100 + d, "texts": [name]} for i, name in enumerate(names) for d in (0, 2)
        ]
        screens[video_id] = {"frames": frames}
    labels, _ = L.label_videos(pd.DataFrame(rows, columns=C.VIDEO_COLUMNS), {}, VOCAB, screens)
    return labels


def commons(labels: list[L.VideoLabel]) -> dict[tuple[str, str], bool]:
    return {(v.video_id, m.name): m.common for v in labels for m in v.exercises}


def test_같은_재생목록_세_편에_똑같이_나오는_이름은_공통이다() -> None:
    labels = build(
        {
            "a": ("꽃게처럼 걸어요", ["엉금엉금 기어요", "거북이 스트레칭"]),
            "b": ("수박나라로 떠나요", ["거북이 스트레칭"]),
            "c": ("체력왕이 되어요", ["거북이 스트레칭"]),
        }
    )
    assert commons(labels) == {
        ("a", "엉금엉금 기어요"): False,
        ("a", "거북이 스트레칭"): True,
        ("b", "거북이 스트레칭"): True,
        ("c", "거북이 스트레칭"): True,
    }
    row = L.labels_frame(labels).iloc[0]
    assert (row["exercise_names"], row["common_exercise_names"]) == (
        "엉금엉금 기어요",
        "거북이 스트레칭",
    )


def test_두_편에만_나오면_공통이_아니다() -> None:
    """실측에서 다음으로 많이 겹친 이름(`엉금엉금 기어요`)은 2편이고 본운동이다."""
    labels = build(
        {
            "a": ("대근육 발달 운동", ["엉금엉금 기어요"]),
            "b": ("근력 운동 응용", ["엉금엉금 기어요"]),
        }
    )
    assert not any(commons(labels).values())


def test_제목에서_잡힌_이름은_공통이_아니다() -> None:
    """`EP03.거북이 스트레칭` 에서는 그것이 본 내용이다."""
    labels = build(
        {
            "ep03": ("EP03.거북이 스트레칭", ["거북이 스트레칭"]),
            "b": ("수박나라로 떠나요", ["거북이 스트레칭"]),
            "c": ("체력왕이 되어요", ["거북이 스트레칭"]),
        }
    )
    assert commons(labels) == {
        ("ep03", "거북이 스트레칭"): False,
        ("b", "거북이 스트레칭"): True,
        ("c", "거북이 스트레칭"): True,
    }


def test_운동_이름_표는_본운동_행을_공통_행보다_먼저_적는다() -> None:
    labels = build(
        {
            "b": ("수박나라로 떠나요", ["거북이 스트레칭"]),
            "c": ("체력왕이 되어요", ["거북이 스트레칭"]),
            "ep03": ("EP03.거북이 스트레칭", ["거북이 스트레칭"]),
        }
    )
    frame = L.exercises_frame(labels)
    assert list(zip(frame["video_id"], frame["common"], strict=True)) == [
        ("ep03", False),
        ("b", True),
        ("c", True),
    ]
