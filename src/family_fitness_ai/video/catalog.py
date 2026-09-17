"""클립 목록과 한 회분 짜기.

영상 하나에 운동이 여럿이라 영상을 통째로 내보내면 아이가 10분을 기다린다.
클립(시작·끝이 있는 한 동작)으로 끊어 두고, 필요한 분량만큼 골라 **준비 → 본 →
정리** 순서로 쌓는다. 클립 하나는 대개 1분 안쪽이라, 15분 한 회는 여러 클립이
모여 만들어진다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache

from family_fitness_ai.common.settings import settings
from family_fitness_ai.rag.index import Chunk, corpus
from family_fitness_ai.video.vocabulary import exercises_in

PHASES = ("준비운동", "본운동", "정리운동")
#: 한 회를 단계에 나누는 몫. 몸을 덥히고, 하고, 푼다.
PHASE_SHARE = {"준비운동": 0.2, "본운동": 0.6, "정리운동": 0.2}
#: 한 단계에 이보다 많이 넣지 않는다. 카드가 길어지면 아이가 안 본다.
MAX_PER_PHASE = 6


@dataclass(frozen=True)
class Clip:
    video_id: str
    name: str
    #: 처방 어휘로 옮긴 이름. 못 옮긴 클립은 빈 문자열이다.
    exercise_name: str
    fitness_factor: str
    phase: str
    start_sec: int
    end_sec: int
    age_group: str
    quiet: bool
    home_ok: bool
    needs_props: bool

    @property
    def duration_sec(self) -> int:
        return self.end_sec - self.start_sec

    @property
    def title(self) -> str:
        """화면에 나가는 이름. 처방 어휘가 있으면 그쪽이 또렷하다."""
        return self.exercise_name or self.name

    def as_video(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
        }


@lru_cache
def clips() -> tuple[Clip, ...]:
    release = settings().release_dir
    labels: dict[str, dict[str, str]] = {}
    with (release / "clip_labels.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            labels[row["name_on_video"]] = row

    videos = {
        chunk.chunk_id.split(":", 1)[1]: chunk
        for chunk in corpus().chunks
        if chunk.source == "video"
    }

    out: list[Clip] = []
    with (release / "video_clips.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            label = labels.get(row["name_on_video"], {})
            if label.get("is_exercise") != "True":
                continue
            video = videos.get(row["video_id"])
            out.append(
                Clip(
                    video_id=row["video_id"],
                    name=row["name_on_video"],
                    exercise_name=label.get("exercise_name", ""),
                    fitness_factor=label.get("fitness_factor", ""),
                    phase=label.get("phase") or row["phase_on_video"] or "본운동",
                    start_sec=int(row["start_sec"]),
                    end_sec=int(row["end_sec"]),
                    age_group=video.age_group if video else "",
                    quiet=label.get("quiet") == "True",
                    home_ok=label.get("home_ok") == "True",
                    needs_props=label.get("needs_props") == "True",
                )
            )
    return tuple(out)


def citation_for(video_id: str) -> Chunk | None:
    return corpus().by_id(f"video:{video_id}")


@dataclass(frozen=True)
class Conditions:
    """부모가 고른 조건. 고른 것만 좁히고 나머지는 건드리지 않는다."""

    quiet: bool = False
    small_space: bool = False
    no_props: bool = False


def _fits(clip: Clip, conditions: Conditions) -> bool:
    if conditions.quiet and not clip.quiet:
        return False
    if conditions.small_space and not clip.home_ok:
        return False
    return not (conditions.no_props and clip.needs_props)


def _rank(clip: Clip, factor: str, prescribed: set[str]) -> tuple[int, int]:
    """앞에 설 차례. 처방에 실제로 나온 동작이 먼저다."""
    score = 0
    if clip.exercise_name and clip.exercise_name in prescribed:
        score -= 4
    if factor and clip.fitness_factor == factor:
        score -= 2
    if clip.exercise_name:
        score -= 1
    return score, -clip.duration_sec


def prescribed_names(chunks: list[Chunk]) -> set[str]:
    names: set[str] = set()
    for chunk in chunks:
        names |= {name for name, _ in exercises_in(chunk.text)}
    return names


def routine(
    age_group: str,
    minutes: int,
    *,
    factor: str = "",
    prescribed: set[str] | None = None,
    conditions: Conditions | None = None,
    exclude: set[str] | None = None,
) -> dict[str, list[Clip]]:
    """한 회분 클립을 단계별로 고른다.

    같은 동작을 두 번 넣지 않고, 한 단계의 몫을 채우면 다음 단계로 넘어간다.
    조건에 걸려 남는 클립이 없으면 그 단계는 빈 채로 둔다 — 조건을 몰래 풀어
    아무거나 채우지 않는다.
    """
    conditions = conditions or Conditions()
    prescribed = prescribed or set()
    pool = [clip for clip in clips() if clip.age_group == age_group and _fits(clip, conditions)]

    picked: dict[str, list[Clip]] = {phase: [] for phase in PHASES}
    # 날마다 같은 차림을 내지 않으려고, 앞선 날에 쓴 동작을 빼고 고른다.
    used: set[str] = set(exclude or ())
    for phase in PHASES:
        budget = minutes * 60 * PHASE_SHARE[phase]
        spent = 0
        # 몫 안에 들어오는 클립이 하나라도 있으면 그것들로만 채운다. 한 편이
        # 통째로 한 동작인 긴 영상 때문에 10분 요청이 14분이 되는 일을 막는다.
        fitting = [c for c in pool if c.phase == phase and c.duration_sec <= max(budget * 1.3, 90)]
        candidates = fitting or [c for c in pool if c.phase == phase]
        if all(c.title in used for c in candidates):
            # 뺄 것을 빼고 나니 남는 게 없다. 그 단계만 처음으로 되돌린다.
            used -= {c.title for c in candidates}
        for clip in sorted(
            candidates,
            key=lambda c: _rank(c, factor, prescribed),
        ):
            if clip.title in used or len(picked[phase]) >= MAX_PER_PHASE:
                continue
            # 이미 한 개라도 담았는데 이 클립이 몫을 크게 넘기면 건너뛴다.
            if picked[phase] and spent + clip.duration_sec > budget * 1.3:
                continue
            picked[phase].append(clip)
            used.add(clip.title)
            spent += clip.duration_sec
            if spent >= budget:
                break
    return picked


def pool(
    age_group: str,
    *,
    conditions: Conditions | None = None,
    factor: str = "",
    prescribed: set[str] | None = None,
    limit: int = 120,
) -> tuple[list[Clip], str]:
    """고를 만한 클립과, 또래 밖까지 갔는지 알리는 말.

    또래 라벨이 맞는 클립을 앞세우되, 거기서 끊지 않는다. 영상에 연령 라벨이
    붙은 편이 많지 않아 또래만 고집하면 여덟 살에게 아무것도 못 준다. 라벨이
    다른 클립도 뒤에 세워 두고, 그런 것이 섞였으면 그 사실을 말로 돌려준다 —
    쓸지 말지는 화면 저쪽에서 정한다.
    """
    conditions = conditions or Conditions()
    prescribed = prescribed or set()
    fitting = [clip for clip in clips() if _fits(clip, conditions)]

    same = [clip for clip in fitting if clip.age_group == age_group]
    other = [clip for clip in fitting if clip.age_group != age_group]
    rank = lambda clip: _rank(clip, factor, prescribed)  # noqa: E731
    same.sort(key=rank)
    other.sort(key=rank)

    picked = same[:limit]
    notice = ""
    if len(picked) < limit // 2:
        room = limit - len(picked)
        picked += other[:room]
        if room and other:
            notice = (
                f"{age_group} 라벨이 붙은 영상이 적어 다른 연령대 영상도 함께 골랐습니다. "
                "그대로 쓸지는 보고 정해 주세요."
            )
    return picked, notice
