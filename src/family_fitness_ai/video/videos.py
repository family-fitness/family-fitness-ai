"""운동영상 찾기. 유튜브 영상과 공단 영상을 한 요청에서 함께 찾는다.

유튜브는 코퍼스의 영상 청크를 임베딩으로 찾는다. 청크 본문에 그 영상에 나오는
운동 이름과 시각이 적혀 있어, 그걸 다시 읽어 구간을 뽑는다 — 유튜브에 다시 가지
않는다. 공단 영상은 코퍼스에 없고 클립 표(kspo_videos.csv)에 있어, 운동명·체력요인이
맞는 클립을 표에서 고른다. 둘을 점수로 한 줄에 세운다.

연령대는 **거르는 조건**이고, 운동명·체력요인은 **찾는 말**이다. 연령 라벨이
없는 영상은 아이 앞에 내지 않는다. 그래서 아이 프로필 검색은 자주 빈다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from family_fitness_ai.rag.index import Chunk
from family_fitness_ai.rag.search import search
from family_fitness_ai.video import catalog

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
    # 어르신은 성인 영상도 또래로 친다(catalog.ages_for).
    result = search(query, k=k, sources=("video",), age_group=catalog.ages_for(age_group))

    ends = {(c.video_id, c.start_sec): c.end_sec for c in catalog.clips() if c.source == "youtube"}
    hits: list[dict[str, Any]] = []
    for hit in result.hits:
        matched, start = match_clip(hit.chunk, exercise_names)
        video_id = hit.chunk.chunk_id.split(":", 1)[1]
        hits.append(
            {
                "source": "youtube",
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}&t={start}s",
                "start_sec": start,
                # 클립 표에 그 시각에서 시작하는 클립이 있으면 그 끝. 없으면 모른다.
                "end_sec": ends.get((video_id, start)),
                "score": hit.score,
                "matched_exercise_names": matched,
                "citation": {
                    "label": hit.chunk.citation_label,
                    "chunk_id": hit.chunk.chunk_id,
                },
            }
        )
    hits += kspo_hits(age_group, fitness_factors, exercise_names)
    hits.sort(key=lambda h: (-h["score"], h["source"], h["video_id"], h["start_sec"]))
    filtered = {
        key: value
        for key, value in result.filtered_out.items()
        if key in ("age_group", "below_threshold")
    }
    return {"hits": hits[:k], "filtered_out": filtered}


#: 공단 클립의 점수. 임베딩 유사도가 아니라 얼마나 맞았는지다 — 이름과 요인이 다
#: 맞으면 0.9, 이름만 0.8, 요인만 0.6. 이름이 맞은 클립이 유튜브 유사도(대개 0.6~0.75)
#: 보다 앞선다.
KSPO_BOTH, KSPO_NAME, KSPO_FACTOR = 0.9, 0.8, 0.6


def kspo_hits(
    age_group: str,
    fitness_factors: tuple[str, ...] = (),
    exercise_names: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """공단 클립 중 맞는 것. 영상마다 하나만 — 표는 한 영상을 요인·단계마다 한 줄씩
    두어서, 그대로 두면 같은 영상이 결과에 여러 번 나온다."""
    best: dict[str, tuple[tuple[float, int, int], dict[str, Any]]] = {}
    ages = catalog.ages_for(age_group)
    for clip in catalog.clips():
        if clip.source != "kspo" or clip.age_group not in ages:
            continue
        names = {_key(clip.title), _key(clip.name)}
        matched = [
            wanted
            for wanted in exercise_names
            if (want := _key(wanted)) and any(want in have or have in want for have in names)
        ]
        by_factor = bool(fitness_factors) and clip.fitness_factor in fitness_factors
        if not matched and not by_factor:
            continue
        score = KSPO_BOTH if matched and by_factor else KSPO_NAME if matched else KSPO_FACTOR
        # 같은 점수면 한 세트로 삼기 좋은 길이, 그다음 앞쪽 클립.
        rank = (score, -abs(clip.duration_sec - catalog.SET_SECONDS), -clip.start_sec)
        if clip.video_id in best and best[clip.video_id][0] >= rank:
            continue
        chunk = catalog.citation_for(clip.video_id)
        best[clip.video_id] = (
            rank,
            {
                "source": "kspo",
                "video_id": clip.video_id,
                "url": clip.url,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "score": score,
                "matched_exercise_names": matched,
                "citation": {
                    "label": chunk.citation_label if chunk else clip.citation_label,
                    "chunk_id": chunk.chunk_id if chunk else f"kspo:{clip.video_id}",
                },
            },
        )
    return [hit for _, hit in best.values()]


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
