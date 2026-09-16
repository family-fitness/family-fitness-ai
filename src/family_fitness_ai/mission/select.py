"""후보를 뽑고 고른다 (docs/05).

**무엇을 추천할지는 코드가 데이터로 정한다** (AGENTS.md §7). 고르는 값은 처방
빈도 하나다 — 그 칸에서 많이 처방된 순서다. 흔한 방식을 먼저 쓴다.

**칸 안에서만 고른다.** 청크는 칸·단계당 `MAX_EXERCISES` 개로 잘리므로
(`cells.chunk_prefix`), 잘린 뒤의 이름을 고르면 인용한 청크가 그 운동을 말하지
않는다. 후보는 청크에 들어간 앞 k개뿐이다.

영상은 **딸린 값**이다. 붙으면 먼저 고르고, 없으면 `video: null` 로 나간다 —
운동명과 근거만으로 카드가 성립해야 한다 (docs/03 §5.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..common.types import CHILD_AGE_GROUPS, AgeGroup, AgeUnit
from ..rag.prescription import identity
from .cells import Cell, CellMatch, chunk_prefix
from .exercises import BUNDLE
from .segments import VideoSegment

if TYPE_CHECKING:  # `build` 가 이 모듈을 쓰지 않으므로 순환은 없으나 한 방향만 둔다
    from .build import Mission as MissionRow

# 만 7~10세 예외의 나이 범위 (docs/02 §2.3).
#
# 그 구간은 측정도 영상도 비어 있어, 아이 연령대 영상을 연령대와 무관하게 내보낸다.
# **연령 미상·성인·어르신 영상은 그래도 내보내지 않는다.**
GAP_AGES = (7, 10)


def video_allowed(
    age_group: AgeGroup | str,
    age: int,
    age_unit: AgeUnit | str,
    video_age_group: str,
) -> bool:
    """이 프로필에게 이 영상을 내보내도 되나 (docs/02 §2.3).

    백엔드도 같은 것을 막는다 — `AgeRange.of(연령대).intersects(age_from, age_to)`
    이고 다섯 구간이 서로 겹치지 않으므로 **연령대가 같을 때만 통과**한다
    (`ExerciseVideo.kt:17-24·43-47`). 만 7~10세 예외는 우리 쪽 규약이고,
    `ProposalConverter` 는 우리가 낸 영상을 연령으로 걸러내지 않으므로
    (카탈로그에 있는지만 본다) 이 판단이 곧 서비스의 판단이다.
    """
    if not video_age_group:
        # 연령 미상. 아이에게는 내보내지 않는다 — 다른 것이 아니라 모르는 것이다
        return age_group not in CHILD_AGE_GROUPS
    if age_group in CHILD_AGE_GROUPS and video_age_group not in CHILD_AGE_GROUPS:
        return False
    if age_unit == "세" and GAP_AGES[0] <= age <= GAP_AGES[1]:
        return video_age_group in CHILD_AGE_GROUPS
    return video_age_group == age_group


@dataclass(frozen=True)
class Candidate:
    """미션 하나가 될 수 있는 것. 근거(`chunk_id`)를 늘 들고 다닌다."""

    exercise_name: str
    phase: str
    # 그 칸·단계에서 몇 번째로 많이 처방됐나 (1부터)
    rank: int
    count: int
    chunk_id: str
    citation_label: str
    segment: VideoSegment | None = None

    @property
    def has_video(self) -> bool:
        return self.segment is not None

    @property
    def common(self) -> bool:
        """공통 준비·마무리 구간인가. 영상이 없으면 공통 여부를 모른다."""
        return self.segment.common if self.segment else False

    @property
    def bundle(self) -> bool:
        """이름이 운동 하나가 아니라 묶음을 가리키나 (`맨몸운동  루틴프로그램`).

        받은 사람이 무엇을 할지 알 수 없어 **미션 제목이 될 수 없다.** 버리지는
        않는다 — 처방 횟수가 가장 많은 이름들이라 버리면 후보가 얇아진다. 순위만
        낮춘다 (운동 표의 `kind` 와 같은 판정이다).
        """
        return bool(BUNDLE.search(self.exercise_name))


def _by_name(segments: list[VideoSegment]) -> dict[str, list[VideoSegment]]:
    out: dict[str, list[VideoSegment]] = {}
    for s in segments:
        out.setdefault(identity(s.exercise_name), []).append(s)
    return out


def _pick_segment(
    options: list[VideoSegment],
    age_group: AgeGroup | str,
    age: int,
    age_unit: AgeUnit | str,
) -> VideoSegment | None:
    """쓸 수 있는 구간 중 하나. **비공통 → 같은 연령대 → 짧은 gap** 순이다.

    `gap` 이 작다는 것은 이름표가 다음 운동 직전까지 떠 있었다는 뜻이고, 그만큼
    그 구간이 그 운동의 것이라는 근거가 된다 (docs/05 §3.2).
    """
    usable = [s for s in options if video_allowed(age_group, age, age_unit, s.age_group)]
    if not usable:
        return None
    return min(
        usable,
        key=lambda s: (s.common, s.age_group != age_group, s.gap_sec, s.video_id, s.start_sec),
    )


def candidates(
    match: CellMatch,
    segments: list[VideoSegment],
    *,
    age: int,
    age_unit: AgeUnit | str,
) -> list[Candidate]:
    """그 칸의 후보 전부. **청크에 들어간 앞 k개만이다.**

    빈 칸이면 빈 목록이다 — 예외를 던지지 않는다. 호출자가 인용 0 으로 거부한다.
    """
    if not match.found or match.age_group is None:
        return []
    index = _by_name(segments)
    out: list[Candidate] = []
    for cell in match.cells:
        for rank, (name, count) in enumerate(
            zip(
                cell.exercise_names[: chunk_prefix(cell)],
                cell.exercise_counts[: chunk_prefix(cell)],
                strict=True,
            ),
            start=1,
        ):
            out.append(
                Candidate(
                    exercise_name=name,
                    phase=cell.phase,
                    rank=rank,
                    count=count,
                    chunk_id=cell.chunk_id,
                    citation_label=cell.citation_label,
                    segment=_pick_segment(
                        index.get(identity(name), []), match.age_group, age, age_unit
                    ),
                )
            )
    return out


def from_missions(
    missions: list[MissionRow],
    match: CellMatch,
    *,
    age: int,
    age_unit: AgeUnit | str,
    sex: str,
) -> list[Candidate]:
    """**미션 집합에서 찾는다** (`missions.csv`).

    이것이 추천의 기본 경로다 — 요청마다 처방 칸에서 조립하지 않고, 이미 만들어 둔
    미션 표를 대상 라벨로 좁힌다. 프론트의 `[미션 추천]` 카드 한 장이 이 표의 한 행이다.

    좁히는 것 둘 —
    - **대상**: `연령대-성별` 이 `targets` 에 있는 미션만. 없으면 그 사람의 또래
      처방에 나온 적이 없다는 뜻이고, 인용 문구가 거짓이 된다
    - **영상 연령**: 그 영상이 이 프로필에게 나갈 수 있나 (`video_allowed`).
      못 나가는 영상이면 **미션을 버리지 않고 영상만 뗀다** — 운동명과 근거만으로
      카드가 성립해야 한다 (docs/03 §5.7)

    **인용은 미션 표에서 가져오지 않고 `match` 에서 가져온다.** 표의 `cell_chunk_id`
    는 그 운동이 가장 앞섰던 칸이라 요청자와 연령대가 다를 수 있다 — 실측에서 처방
    인용 12건 중 **6건**이 어긋났고, 유소년 11세 아이에게 「성인 62세 처방」이
    근거로 붙었다 (2026-09-16). 근거는 **묻는 사람의 칸**이어야 한다.

    그래서 그 사람의 칸에 실제로 있는 운동만 남는다 — 표의 `targets` 가 통과시켜도
    청크 안에 이름이 없으면 인용이 그 운동을 말하지 못한다.
    """
    if not match.found or match.age_group is None:
        return []
    wanted = f"{match.age_group}-{sex}"
    # 그 사람의 칸에서 (운동 → 단계·순위·횟수·근거) 를 만든다. 청크에 든 앞 k개만이다
    mine: dict[str, tuple[str, int, int, str, str]] = {}
    for cell in match.cells:
        keep = chunk_prefix(cell)
        for rank, (name, count) in enumerate(
            zip(cell.exercise_names[:keep], cell.exercise_counts[:keep], strict=True), start=1
        ):
            key = identity(name)
            if key not in mine or rank < mine[key][1]:
                mine[key] = (cell.phase, rank, count, cell.chunk_id, cell.citation_label)

    out: list[Candidate] = []
    for mission in missions:
        if wanted not in mission.targets:
            continue
        found = mine.get(identity(mission.exercise_name))
        if found is None:
            # 표는 통과시켰지만 이 사람의 칸 청크에는 없다 — 근거를 댈 수 없다
            continue
        phase, rank, count, chunk_id, label = found
        segment = mission.segment
        if segment is not None and not video_allowed(
            match.age_group, age, age_unit, segment.age_group
        ):
            segment = None
        out.append(
            Candidate(
                exercise_name=mission.exercise_name,
                phase=phase,
                rank=rank,
                count=count,
                chunk_id=chunk_id,
                citation_label=label,
                segment=segment,
            )
        )
    return out


def in_evidence(cell: Cell, exercise_name: str) -> bool:
    """고른 운동이 **인용한 청크 본문 안에** 있나.

    없으면 근거가 그 운동을 말하지 않는 인용이 된다. `candidates` 가 이미
    앞 k개만 내지만, 편성이 후보를 다른 데서 받아 올 수도 있어 따로 둔다.
    """
    wanted = identity(exercise_name)
    return any(identity(n) == wanted for n in cell.exercise_names[: chunk_prefix(cell)])


# 고르는 단계. 본운동이 그 사람이 할 운동이고, 준비·정리는 그 앞뒤다.
MAIN_PHASE = "본운동"


def select(
    pool: list[Candidate],
    count: int,
    *,
    rotate: int = 0,
) -> list[Candidate]:
    """서로 다른 운동 `count` 개.

    순서는 **영상 있음 → 공통 아님 → 묶음 아님 → 본운동 → 처방 순위**다.

    `rotate` 는 같은 요청이 주마다 다른 조합을 내게 한다 (ISO 주차를 넣는다).
    **정렬은 결정적이다** — 같은 `rotate` 면 같은 편성이다.

    `본운동` 을 먼저 쓴다. 다 쓰면 다른 단계로 넘어간다 — 세 개를 달라고 했는데
    두 개를 내지 않는다.
    """
    if count <= 0:
        return []

    picked: list[Candidate] = []
    seen: set[str] = set()
    # **영상이 첫 축이다.** 백엔드는 미션 대부분에 영상이 붙기를 바라고, 영상 붙은
    # 후보는 연령대마다 손에 꼽는다 (유소년 1·청소년 6·성인 0). 단계를 앞세우면
    # 준비·정리운동에 있는 영상 붙은 후보가 영상 없는 본운동에 밀린다 — 실제로
    # 유소년 11세의 영상 후보 2개가 그렇게 빠졌다.
    #
    # 순서 — 영상 있음 → 공통 아님 → 묶음 아님 → 본운동 → 처방 순위.
    for want_video in (True, False):
        for want_plain in (True, False):
            for want_single in (True, False):
                for phase_first in (True, False):
                    tier = [
                        c
                        for c in pool
                        if c.has_video is want_video
                        and (not c.common) is want_plain
                        and (not c.bundle) is want_single
                        and (c.phase == MAIN_PHASE) is phase_first
                    ]
                    # **회전은 순위 묶음 안에서만 한다.** 묶음을 넘어 돌리면 영상
                    # 있는 후보를 제치고 없는 것이 먼저 나온다 — 우선순위가 뒤집힌다.
                    tier.sort(key=lambda c: (c.rank, c.exercise_name))
                    if rotate and tier:
                        offset = rotate % len(tier)
                        tier = tier[offset:] + tier[:offset]
                    for c in tier:
                        key = identity(c.exercise_name)
                        if key in seen:
                            continue
                        seen.add(key)
                        picked.append(c)
                        if len(picked) == count:
                            return picked
    return picked
