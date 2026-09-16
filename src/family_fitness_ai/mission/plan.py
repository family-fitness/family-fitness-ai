"""일일·주간 미션을 짠다 (docs/05).

**일일과 주간은 기간으로만 가른다** (2026-09-16 결정). 주기를 담는 필드를 새로
만들지 않는다 — 미션 하나의 `period` 가 하루면 일일, 이레면 주간이다.

    일일 미션   period.start_date == end_date · 세션 1개 · day_offset 0
    주간 미션   start_date ~ +6일 · 세션 days_per_week 개

**주간 미션은 그 주 일일 미션의 합이다.** 운동을 다르게 두면 아이가 일일 세 개만
해도 주간이 완료로 뜬다 — `TIMER_MINUTES` 가 기간 안 활동 합계라서다
(`MissionCompletionPolicy`). 같은 운동을 담으면 "일일을 다 하면 주간도 된다"가
참이 되고, 실제로 하지 않은 것을 완료로 보고하는 자리가 없다.

> **카드의 분을 더하면 실제의 두 배다.** 일일 3×15분과 주간 45분이 같은 45분을
> 가리킨다. 프론트가 카드를 합산하지 않아야 한다.

**무엇을 추천할지는 `select` 가 미션 집합에서 정하고, LLM 은 카드 문구만 쓴다**
(AGENTS.md §7). `writer` 를 주지 않으면 규칙 문구로 나가고 외부 호출이 0이다 —
LLM 이 실패하거나 금지 어휘를 쓰면 규칙 문구로 **강등**한다 (docs/01 §3.1).

프론트의 `[미션 추천]` 카드가 `운동 이름 + 영상 링크 + 짧은 문구`이고, 그 문구가
`copy.child`·`copy.parent` 다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..api.coach_schemas import (
    Citation,
    Mission,
    MissionCopy,
    MissionPeriod,
    Participant,
    Proposal,
    Session,
    Video,
)
from ..coach.llm import (
    MISSION_SYSTEM,
    AnswerWriter,
    mission_prompt,
    parse_copy,
    weekly_claim_ok,
)
from ..common.copy import contains_forbidden
from ..common.types import resolve_age_group
from .build import Mission as MissionRow
from .cells import Cell, CellMatch, find_cells
from .select import Candidate, from_missions, select

# docs/03 §5.1 · 백엔드 `CoachRoles`. `응원` 은 편성에서 빠지고 참여자에도 넣지 않는다
# (`LabelBasedProposalPlanner` 가 `role != CHEER` 로 거른다).
DRIVER = "주행자"
COMPANION = "동반자"
CHEER = "응원"

WEEK_DAYS = 7

# 처방 순위를 문구에 쓸 상한. 28번째를 「28번째로 많이 나온」이라 적으면 근거가 아니라
# 군더더기다 — 상위일 때만 순위를 말하고 나머지는 「처방에 나온」으로 둔다.
RANK_IN_COPY = 5


class Refusal(Exception):
    """인용을 만들 수 없다 → 거부 (docs/03 §5.4). 사유는 계약의 값이다."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass
class _Citations:
    """인용 번호를 나눠 준다. 같은 근거는 같은 번호다."""

    items: list[Citation] = field(default_factory=list)
    _index: dict[tuple[str, str | None], int] = field(default_factory=dict)

    def add(self, label: str, chunk_id: str, url: str | None = None) -> int:
        key = (chunk_id, url)
        if key in self._index:
            return self._index[key]
        number = len(self.items) + 1
        self.items.append(Citation(index=number, label=label, chunk_id=chunk_id, url=url))
        self._index[key] = number
        return number


@dataclass(frozen=True)
class MemberPlan:
    """한 사람 몫. `daily` 가 비면 주간만 낸다."""

    ref: str
    role: str
    age_group: str
    match: CellMatch
    pool: tuple[Candidate, ...]
    picks: tuple[Candidate, ...]
    daily: bool

    @property
    def pulled_note(self) -> str:
        """칸이 없어 나이를 당겼으면 그 사실. `retrieve` 요약에 남는다 — 인용 문구의
        나이가 요청한 나이와 다른 이유가 기록에 있어야 한다."""
        if not self.match.pulled:
            return ""
        return f"{self.match.pulled_from}→{self.match.pulled_to} 당김"


def is_daily(mission: Mission) -> bool:
    """일일 미션인가. **기간이 하루면 일일이다**.

    주기를 담는 필드가 없으므로 이 판정이 규칙의 정본이다. 라우터·시험·문서가
    같은 함수를 본다 — 세 곳에 따로 쓰면 갈라진다.
    """
    return mission.period.start_date == mission.period.end_date


def count_kinds(missions: list[Mission]) -> tuple[int, int]:
    """(일일, 주간) 개수."""
    daily = sum(1 for m in missions if is_daily(m))
    return daily, len(missions) - daily


def day_offsets(days_per_week: int) -> list[int]:
    """이레에 `days_per_week` 번을 고르게 편다. 3이면 0·2·4 다.

    백엔드 대체 편성과 같은 계산이다 (`LabelBasedProposalPlanner`) — 두 경로가
    다른 날짜를 내면 어느 것이 맞는지가 흐려진다.
    """
    return [(i * WEEK_DAYS) // days_per_week for i in range(days_per_week)]


def _sex_word(age_group: str, sex: str) -> str:
    """인용 문구에 쓸 말. 아이에게 「남성」이라 쓰지 않는다."""
    child = age_group in ("유아기", "유소년", "청소년")
    if sex == "M":
        return "남아" if child else "남성"
    if sex == "F":
        return "여아" if child else "여성"
    return ""


def _child_copy(member: MemberPlan, picks: tuple[Candidate, ...], daily: bool) -> str:
    """아이 화면 문구. **아이가 참여하지 않는 미션에는 비워 둔다.**

    동반자만 들어간 미션의 문구를 아이 화면에 띄우면 아이가 자기 것으로 읽는다.
    계약은 빈 문자열을 허용한다 (docs/03 §5.6 · `CopyBody.child` 기본값 `""`).
    """
    if member.role != DRIVER:
        return ""
    if daily:
        return f"오늘은 {picks[0].exercise_name} 한 번 해볼까요"
    return f"이번 주는 {len(picks)}가지 동작을 나누어 해볼까요"


def _who(member: MemberPlan) -> str:
    # 정확한 나이는 인용 문구에 이미 있다 (「… 운동처방 · 유소년 11세」). 여기서
    # 다시 적으면 당김이 일어난 경우 두 값이 갈린다.
    word = _sex_word(member.age_group, member.match.sex or "")
    return f"또래 {member.match.age_group} {word}".rstrip()


def _parent_daily(member: MemberPlan, pick: Candidate, minutes: int) -> str:
    """일일 미션의 부모 문구. **그 운동이 무엇을 기른다고 말하지 않는다.**

    요인은 재지 않았다 (docs/05 — 처방 역산을 요인의 출처로 쓰지
    않는다). 말할 수 있는 것은 「또래 처방에 이만큼 나왔다」뿐이라, 순위가 상위면
    순위를, 아니면 **횟수**를 적는다. 둘 다 잰 값이다.
    """
    seen = f"{pick.rank}번째로 많이 나온" if pick.rank <= RANK_IN_COPY else f"{pick.count:,}회 나온"
    return f"{_who(member)} 처방에 {seen} {pick.phase}입니다. {minutes}분이면 충분합니다"


def _parent_weekly(member: MemberPlan, picks: tuple[Candidate, ...], minutes: int) -> str:
    """주간 미션의 부모 문구. 첫 운동만 설명하지 않는다 — 셋을 다 담은 미션이다."""
    times = len(picks)
    return (
        f"{_who(member)} 처방에 많이 나온 순서로 {times}가지를 넣었습니다. "
        f"주 {times}회 {minutes}분이면 충분합니다"
    )


def _session(
    pick: Candidate,
    day_offset: int,
    minutes: int,
    cites: _Citations,
) -> Session:
    evidence = [cites.add(pick.citation_label, pick.chunk_id)]
    video = None
    if pick.segment is not None:
        seg = pick.segment
        evidence.append(cites.add(seg.citation_label, seg.chunk_id, seg.url))
        video = Video(video_id=seg.video_id, start_sec=seg.start_sec)
    return Session(
        day_offset=day_offset,
        exercise_name=pick.exercise_name,
        # 재지 않은 것을 계약 필드에 넣지 않는다 (docs/05).
        fitness_factor="",
        duration_min=minutes,
        video=video,
        evidence=evidence,
    )


def _copy_or_rules(
    member: MemberPlan,
    picks: tuple[Candidate, ...],
    minutes: int,
    *,
    daily: bool,
    writer: AnswerWriter | None,
) -> MissionCopy:
    """LLM 문구를 먼저 쓰고, 없거나 걸리면 규칙 문구로 내려앉는다."""
    if written := _llm_copy(member, picks, minutes, daily, writer, len(member.picks)):
        return MissionCopy(child=written[0], parent=written[1])
    parent = (
        _parent_daily(member, picks[0], minutes)
        if daily
        else _parent_weekly(member, picks, minutes)
    )
    return MissionCopy(child=_child_copy(member, picks, daily), parent=parent)


def _title(picks: tuple[Candidate, ...]) -> str:
    """운동 이름만. **「오늘 · 」 같은 접두어를 붙이지 않는다** —
    기간에 이미 있는 값이고 120자에서 잘리면 뒤가 조용히 깨진다."""
    return " · ".join(p.exercise_name for p in picks)


def _llm_copy(
    member: MemberPlan,
    picks: tuple[Candidate, ...],
    minutes: int,
    daily: bool,
    writer: AnswerWriter | None,
    days_per_week: int,
) -> tuple[str, str] | None:
    """LLM 이 쓴 (아이, 보호자) 문구. 쓸 수 없으면 `None` — 호출자가 규칙 문구를 쓴다.

    **금지 어휘가 있으면 버린다** (docs/03 §2.6). 고쳐서 통과시키지 않는다.
    """
    if writer is None:
        return None
    exercises = tuple((p.exercise_name, p.phase, p.rank, p.count) for p in picks)
    try:
        written = writer.write_copy(
            MISSION_SYSTEM,
            mission_prompt(_who(member), exercises, minutes, daily, days_per_week),
        )
        child, parent = parse_copy(written)
    except Exception:
        return None  # 강등이지 거부가 아니다 (docs/01 §3.1)
    if contains_forbidden(child) or contains_forbidden(parent):
        return None
    # 주당 횟수를 지어냈으면 버린다 — 카드에 틀린 수가 실리면 안 된다
    if not all(
        weekly_claim_ok(text, daily=daily, days_per_week=days_per_week) for text in (child, parent)
    ):
        return None
    # 아이가 참여하지 않는 미션의 문구는 아이 화면에 띄우지 않는다
    return ("" if member.role != DRIVER else child), parent


def _member_missions(
    member: MemberPlan,
    start: dt.date,
    minutes: int,
    cites: _Citations,
    writer: AnswerWriter | None = None,
) -> list[Mission]:
    who = [Participant(ref=member.ref, role=member.role)]
    offsets = day_offsets(len(member.picks))
    out: list[Mission] = []

    if member.daily:
        for pick, offset in zip(member.picks, offsets, strict=True):
            day = start + dt.timedelta(days=offset)
            out.append(
                Mission(
                    title=_title((pick,)),
                    period=MissionPeriod(start_date=day, end_date=day),
                    participants=who,
                    # 일일은 세션 1개이고 그 미션 안에서의 날짜차가 0 이다
                    sessions=[_session(pick, 0, minutes, cites)],
                    copy=_copy_or_rules(member, (pick,), minutes, daily=True, writer=writer),
                )
            )

    out.append(
        Mission(
            title=_title(member.picks),
            period=MissionPeriod(
                start_date=start, end_date=start + dt.timedelta(days=WEEK_DAYS - 1)
            ),
            participants=who,
            sessions=[
                _session(pick, offset, minutes, cites)
                for pick, offset in zip(member.picks, offsets, strict=True)
            ],
            copy=_copy_or_rules(member, member.picks, minutes, daily=False, writer=writer),
        )
    )
    return out


def members(
    profiles: list[tuple[str, str, int, str, str]],
    cells: list[Cell],
    mission_set: list[MissionRow],
    *,
    days_per_week: int,
    rotate: int = 0,
) -> list[MemberPlan]:
    """편성 대상마다 한 몫. `profiles` 는 `(ref, role, age, age_unit, sex)` 다.

    `응원` 은 빠진다. 칸이 없거나 후보가 없는 사람도 빠진다 — 근거 없이 미션을
    내지 않는다. 전원이 빠지면 호출자가 거부한다.
    """
    out: list[MemberPlan] = []
    for ref, role, age, age_unit, sex in profiles:
        if role == CHEER:
            continue
        age_group = resolve_age_group(age, age_unit)  # type: ignore[arg-type]
        if age_group is None:
            continue
        match = find_cells(cells, age_group, age, age_unit, sex)
        if not match.found:
            continue
        # **미션 집합에서 찾는다.** 칸(`match`)은 나이 당김과 인용 문구에
        # 쓰고, 후보는 이미 만들어 둔 표에서 대상 라벨로 좁힌다.
        pool = from_missions(mission_set, match, age=age, age_unit=age_unit, sex=sex)
        picks = select(pool, days_per_week, rotate=rotate)
        if not picks:
            continue
        out.append(
            MemberPlan(
                ref=ref,
                role=role,
                age_group=age_group,
                match=match,
                pool=tuple(pool),
                picks=tuple(picks),
                # 2026-09-16 결정 — 아이(주행자)가 일일과 주간을 함께 받는다.
                # 동반자는 주간만 받는다 (자기 연령대 칸에서 뽑은 것이다).
                daily=role == DRIVER,
            )
        )
    return out


def build(
    profiles: list[tuple[str, str, int, str, str]],
    cells: list[Cell],
    mission_set: list[MissionRow],
    *,
    start_date: dt.date,
    days_per_week: int,
    minutes_per_session: int,
    rotate: int = 0,
    writer: AnswerWriter | None = None,
) -> tuple[Proposal, list[MemberPlan]]:
    """제안 하나. 인용이 0 이면 `Refusal` 이다 (docs/03 §5.4).

    `rotate` 로 같은 요청이 주마다 다른 조합을 낸다. ISO 주차를 넣는다.
    """
    planned = members(profiles, cells, mission_set, days_per_week=days_per_week, rotate=rotate)
    if not planned:
        raise Refusal(
            "no_candidate",
            "편성 대상 가운데 처방 칸이 잡히는 사람이 없다",
        )

    cites = _Citations()
    missions: list[Mission] = []
    for member in planned:
        missions += _member_missions(member, start_date, minutes_per_session, cites, writer)

    if not cites.items:
        raise Refusal("no_citation_generated", "인용을 만들 수 없다")
    return Proposal(missions=missions, citations=cites.items), planned


def iso_week(day: dt.date) -> int:
    """회전값. 같은 주에 같은 편성이고 다음 주에 다른 편성이다."""
    return day.isocalendar().week
