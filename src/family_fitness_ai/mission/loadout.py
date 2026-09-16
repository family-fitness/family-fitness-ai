"""백엔드가 적재할 모양으로 접는다 (docs/05).

`exercise_videos` 표는 **영상 1편 = 1행**이다 (`V1__init.sql:170-187`). 우리
`video_segments.csv` 는 구간 1개 = 1행이라, 한 영상의 구간 여럿이 한 줄로 접힌다.
**그 손실은 인터페이스 확정 항목이다** (docs/05) — 여기서 접는 것은
지금 표가 받을 수 있는 모양이 그것뿐이기 때문이고, 우리 산출물은 구간을 그대로 들고 있다.

**우리가 백엔드 저장소를 고치지 않는다** (AGENTS.md §2). 넣을 SQL 을 파일로 내고,
넣는 것은 백엔드 담당이 한다. 운영의 그 표는 지금 비어 있다 — flyway locations 가
`db/migration` 뿐이라 `db/seed` 를 읽지 않는다. **넣기 전에는 영상이 항상 null 이다.**

실행 (mission.segments 뒤):
    python -m family_fitness_ai.mission.loadout
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..common.settings import RELEASE_DIR
from ..labeling.label import LABELING_FILE, watch_url
from .segments import SEGMENTS_FILE, VideoSegment, read_segments

# 열은 `exercise_videos` 표를 그대로
# 쓰지만 파일 이름은 `video_labels.csv` 다 — 라벨링 중간 산출물은 `interim/
# video_labeling.csv` 로 갈라 두어 헷갈리지 않는다.
VIDEOS_FILE = "video_labels.csv"
SQL_FILE = "video_labels.sql"

# `V1__init.sql:170-187` 의 열 이름 그대로다. **줄이거나 이름을 바꾸지 않는다.**
VIDEO_COLUMNS = [
    "video_id",
    "title",
    "url",
    "duration_sec",
    "age_from",
    "age_to",
    "factors",
    "intensity",
    "space",
    "noise",
    "equipment",
    "labeled_by",
    "label_model",
]

# 연령대 → 만 나이 폭. 백엔드 `AgeRange.of` 와 같은 값이다 (`ExerciseVideo.kt:17-24`) —
# 두 벌이 되면 우리가 낸 라벨이 그쪽 필터를 통과하지 못한다.
AGE_RANGE: dict[str, tuple[int, int]] = {
    "유아기": (0, 6),
    "유소년": (7, 12),
    "청소년": (13, 18),
    "성인": (19, 64),
    "어르신": (65, 120),
}

LABELED_BY = "batch"


def _text(row: pd.Series[object] | None, column: str) -> str:
    """빈 칸을 빈 문자열로 읽는다.

    **`str(value or "")` 로는 안 된다** — pandas 의 `NaN` 은 truthy 라서 문자열
    `"nan"` 이 그대로 CSV·SQL 에 들어간다 (2026-09-16 실측). 더미를 넣지 않는다.
    """
    if row is None:
        return ""
    value = row.get(column)
    return "" if value is None or pd.isna(value) else str(value)


def build(
    segments: list[VideoSegment],
    labeling: pd.DataFrame,
) -> pd.DataFrame:
    """영상 1편 = 1행. 구간의 연령대·요인을 영상 하나로 모은다.

    **연령 폭이 여러 연령대에 걸치면 넣지 않는다.** 한 영상의 구간들이 서로 다른
    연령대라면 그 영상의 연령을 하나로 말할 수 없다 — 폭을 넓혀 두 연령대를 덮으면
    백엔드 필터가 양쪽에 내보낸다 (`intersects`). 그런 영상은 연령을 비워 내고,
    비면 아이에게 나가지 않는다 (`suitableFor`).
    """
    meta = labeling.set_index("video_id")
    rows = []
    by_video: dict[str, list[VideoSegment]] = {}
    for segment in segments:
        by_video.setdefault(segment.video_id, []).append(segment)

    for video_id, group in sorted(by_video.items()):
        groups = {s.age_group for s in group if s.age_group}
        age_from, age_to = (None, None)
        if len(groups) == 1:
            age_from, age_to = AGE_RANGE.get(groups.pop(), (None, None))

        row = meta.loc[video_id] if video_id in meta.index else None
        factors = _text(row, "fitness_factors")
        rows.append(
            {
                "video_id": video_id,
                "title": _text(row, "title"),
                # 영상 1행이라 구간 시각을 붙이지 않는다 — 구간별 주소는
                # `video_segments.csv` 에 있다 (계약의 키는 video_id·start_sec 다)
                "url": watch_url(video_id, None),
                "duration_sec": None if row is None else row.get("duration_sec"),
                "age_from": age_from,
                "age_to": age_to,
                # 쉼표로 이은 한글 요인 (varchar(120)) — 우리 산출물은 세미콜론이다
                "factors": ",".join(x for x in factors.split(";") if x),
                # **잰 값이 없다.** 더미로 채우지 않는다
                "intensity": None,
                "space": None,
                "noise": None,
                "equipment": None,
                "labeled_by": LABELED_BY,
                "label_model": _text(row, "labeler_version"),
            }
        )
    frame = pd.DataFrame(rows, columns=VIDEO_COLUMNS)
    for column in ("duration_sec", "age_from", "age_to"):
        frame[column] = frame[column].astype("Int64")  # 1575.0 으로 쓰이지 않게
    return frame


def _sql_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "NULL"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def to_sql(frame: pd.DataFrame) -> str:
    """**멱등이다.** 두 번 돌려도 행 수가 같다 — `ON CONFLICT` 로 갱신한다.

    백엔드 담당이 그대로 붙여 넣을 수 있게 한 문장으로 낸다.
    """
    columns = ", ".join(VIDEO_COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in VIDEO_COLUMNS if c != "video_id")
    values = ",\n  ".join(
        "(" + ", ".join(_sql_value(row[c]) for c in VIDEO_COLUMNS) + ")"
        for _, row in frame.iterrows()
    )
    return (
        "-- family-fitness-ai 가 낸 것이다. 손으로 고치지 말고 다시 내라:\n"
        "--   python -m family_fitness_ai.mission.loadout\n"
        "-- intensity·space·noise·equipment 는 잰 값이 없어 NULL 이다 (더미를 넣지 않는다).\n"
        f"INSERT INTO exercise_videos ({columns})\nVALUES\n  {values}\n"
        f"ON CONFLICT (video_id) DO UPDATE SET {updates};\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="exercise_videos 적재 형식과 멱등 SQL 을 낸다")
    ap.add_argument("--release", default=str(RELEASE_DIR), help="낼 곳 (커밋한다)")
    ap.add_argument("--interim", default="data/interim", help="라벨링 중간 산출물이 있는 곳")
    args = ap.parse_args(argv)

    release, interim = Path(args.release), Path(args.interim)
    segments = read_segments(release / SEGMENTS_FILE)
    if not segments:
        print(f"[중단] 구간이 없다: {release / SEGMENTS_FILE}")
        return 1
    labeling_path = interim / LABELING_FILE
    if not labeling_path.exists():
        print(f"[중단] 라벨링 산출물이 없다: {labeling_path}")
        return 1

    frame = build(segments, pd.read_csv(labeling_path, encoding="utf-8-sig"))
    # utf-8-sig — 검수하는 사람이 엑셀로 연다 (docs/02 §4 ③)
    frame.to_csv(release / VIDEOS_FILE, index=False, encoding="utf-8-sig")
    (release / SQL_FILE).write_text(to_sql(frame), encoding="utf-8")

    aged = int(frame["age_from"].notna().sum())
    print(f"영상 {len(frame)}행 · 연령 라벨 있음 {aged} · 없음 {len(frame) - aged}")
    print(frame.groupby(["age_from", "age_to"], dropna=False).size().to_string())
    print(f"→ {release / VIDEOS_FILE}")
    print(f"→ {release / SQL_FILE}  (백엔드 담당이 넣는다 — 우리가 넣지 않는다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
