"""운동영상 찾기.

영상 청크의 본문에는 그 영상에 나오는 운동 이름과 시각이 적혀 있다. 그걸 다시
읽어 구간을 뽑는다 — 유튜브에 다시 가지 않는다.

연령대는 **거르는 조건**이고, 운동명·체력요인은 **찾는 말**이다. 연령 라벨이
없는 영상은 아이 앞에 내지 않는다. 그래서 아이 프로필 검색은 자주 빈다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from family_fitness_ai.rag.index import Chunk
from family_fitness_ai.rag.search import search

_LISTED = ("나오는 운동:", "준비·마무리:")
#: 「나비자세(02:24)」
_NAMED = re.compile(r"^(?P<name>.+?)\((?P<time>\d{1,2}:\d{2}(?::\d{2})?)\)$")
#: 「00:05 다양한 공을 굴려요」
_CHAPTER = re.compile(r"^(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s+(?P<name>.+)$")


@dataclass(frozen=True)
class Clip:
    name: str
    #: 시각이 안 적힌 운동도 있다. 영상 안에 나오기는 하는데 어디인지 모른다.
    start_sec: int | None


def _seconds(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return parts[0] * 60 + parts[1]


def clips_of(chunk: Chunk) -> list[Clip]:
    clips: list[Clip] = []
    for segment in chunk.text.split(" · "):
        segment = segment.strip()
        listed = next((p for p in _LISTED if segment.startswith(p)), None)
        if listed:
            for piece in segment[len(listed) :].split(","):
                piece = piece.strip()
                if not piece:
                    continue
                found = _NAMED.match(piece)
                if found:
                    clips.append(Clip(found["name"].strip(), _seconds(found["time"])))
                else:
                    clips.append(Clip(piece, None))
            continue
        chapter = _CHAPTER.match(segment)
        if chapter:
            clips.append(Clip(chapter["name"].strip(), _seconds(chapter["time"])))
    return clips


def _key(name: str) -> str:
    return re.sub(r"\s+", "", name)


def match_clip(chunk: Chunk, names: tuple[str, ...]) -> tuple[list[str], int]:
    """요청한 운동명 중 이 영상에 있는 것과, 어디서부터 볼지.

    이름이 맞는 구간이 있으면 그 시각으로, 없으면 0초(처음부터)로 연다.
    """
    clips = clips_of(chunk)
    matched: list[str] = []
    start = None
    for wanted in names:
        want = _key(wanted)
        for clip in clips:
            have = _key(clip.name)
            if want and (want in have or have in want):
                matched.append(wanted)
                if start is None and clip.start_sec is not None:
                    start = clip.start_sec
                break
    if start is None:
        timed = [clip for clip in clips if clip.start_sec is not None]
        start = timed[0].start_sec if timed and matched else 0
    return matched, start or 0


def search_videos(
    age_group: str,
    fitness_factors: tuple[str, ...] = (),
    exercise_names: tuple[str, ...] = (),
    k: int = 5,
) -> dict[str, object]:
    query = " ".join([*exercise_names, *fitness_factors, age_group, "운동 영상"])
    result = search(query, k=k, sources=("video",), age_group=age_group)

    hits = []
    for hit in result.hits:
        matched, start = match_clip(hit.chunk, exercise_names)
        hits.append(
            {
                "video_id": hit.chunk.chunk_id.split(":", 1)[1],
                "start_sec": start,
                "score": hit.score,
                "matched_exercise_names": matched,
                "citation": {
                    "label": hit.chunk.citation_label,
                    "chunk_id": hit.chunk.chunk_id,
                },
            }
        )
    filtered = {
        key: value
        for key, value in result.filtered_out.items()
        if key in ("age_group", "below_threshold")
    }
    return {"hits": hits, "filtered_out": filtered}


def best_video(age_group: str, exercise_name: str, factor: str) -> tuple[Chunk, int] | None:
    """미션 한 칸에 붙일 영상. 없으면 None — 영상 없이도 카드는 선다."""
    result = search(
        f"{exercise_name} {factor} {age_group}",
        k=3,
        sources=("video",),
        age_group=age_group,
    )
    if not result.hits:
        return None
    for hit in result.hits:
        matched, start = match_clip(hit.chunk, (exercise_name,))
        if matched:
            return hit.chunk, start
    top = result.hits[0]
    _, start = match_clip(top.chunk, (exercise_name,))
    return top.chunk, start
