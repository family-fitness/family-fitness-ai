"""한 주 편성.

네 걸음이다 — assess · retrieve · compose · verify. 걸음마다 무엇을 보고 골랐는지
summary 에 남긴다. 나중에 「왜 이 운동이지?」를 되짚을 수 있어야 한다.

무엇이 무엇을 맡는가

    assess    측정값을 또래와 대조한다. 계산이다. LLM 이 끼지 않는다.
    retrieve  나이·성별·요인으로 실제 처방된 운동을 찾는다. 검색이다.
    compose   찾아 둔 클립과 근거를 놓고 **LLM 이 한 주를 짠다.** 클립 목록 밖의
              것은 고를 수 없고, 고른 id 는 돌아온 뒤 다시 확인한다. LLM 이 없거나
              넘어지면 규칙이 대신 짠다 — 서비스가 멈추지는 않는다.
    verify    인용과 문구를 보고 내보낸다.

없으면 없다고 하지 않고, 넓혀서 권하고 넓혔다고 말한다. 여덟 살에게 그 나이
처방 자료가 없다고 빈손으로 돌려보내면 서비스가 아니다. 같은 연령대의 가까운
나이로, 그래도 없으면 연령대 밖으로 넓히고, 넓힌 사실을 notice 로 돌려준다.
쓸지 말지는 화면 저쪽에서 정한다.

AI 는 DB 에 쓰지 않는다. 여기서 나오는 것은 **제안**이고, 저장과 승인은 전부
호출자가 한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from family_fitness_ai.coach import llm as coach_llm
from family_fitness_ai.common import copy as words
from family_fitness_ai.common.items import age_group_of
from family_fitness_ai.rag.index import Chunk
from family_fitness_ai.rag.search import search
from family_fitness_ai.stats.assess import Profile, factor_rows
from family_fitness_ai.video import catalog

#: prescription:유소년-11-F-유연성-1
_CHUNK_ID = re.compile(
    r"^prescription:(?P<age_group>[^-]+)-(?P<age>\d+)-(?P<sex>[MF])-(?P<factor>[^-]+)-(?P<grade>.+)$"
)

#: 주 n 회를 한 주 어디에 놓을까. 이틀 이상 붙여 두지 않는다.
_WEEK_SLOTS = {
    1: (0,),
    2: (0, 3),
    3: (0, 2, 4),
    4: (0, 1, 3, 5),
    5: (0, 1, 2, 3, 4),
    6: (0, 1, 2, 3, 4, 5),
    7: (0, 1, 2, 3, 4, 5, 6),
}
_WEEKDAYS = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")

#: 주간 미션의 요일 자리. 날짜를 박지 않는다 — 그 주 안에 아무 때나 하는 것이다.
WEEKLY_SLOT = -1


@dataclass
class Slot:
    """미션 한 건이 놓일 자리."""

    kind: str  # 일간 · 주간
    offset: int  # 일간이면 그 주의 몇째 날, 주간이면 WEEKLY_SLOT
    minutes: int
    participants: list[dict[str, str]]

    @property
    def weekly(self) -> bool:
        return self.kind == "주간"


#: band → 또래 중 비슷한 처지의 처방 칸. 「나와 비슷한 아이들이 실제로 받은 것」.
_BAND_GRADE = {"growth": "참가", "steady": "2", "strength": "1"}


@dataclass
class RunProfile:
    ref: str
    role: str
    age: int
    age_unit: str
    sex: str
    input_level: str = "L0"
    height_cm: float | None = None
    weight_kg: float | None = None
    measurements: dict[str, float] | None = None

    @property
    def age_group(self) -> str:
        return age_group_of(self.age, self.age_unit)

    def profile(self) -> Profile:
        return Profile(
            profile_ref=self.ref,
            age=self.age,
            age_unit=self.age_unit,
            sex=self.sex,
            height_cm=self.height_cm,
            weight_kg=self.weight_kg,
            measurements=self.measurements,
        )


@dataclass
class Constraints:
    days_per_week: int = 3
    minutes_per_session: int = 15
    #: 한 주에 한 번 길게 하는 회. None 이면 주간 미션을 만들지 않는다.
    weekly_minutes: int | None = None
    quiet: bool = False
    small_space: bool = False
    no_props: bool = False

    def conditions(self) -> catalog.Conditions:
        return catalog.Conditions(
            quiet=self.quiet, small_space=self.small_space, no_props=self.no_props
        )


@dataclass
class Step:
    seq: int
    name: str
    status: str
    summary: str

    def dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
        }


@dataclass
class Plan:
    steps: list[Step]
    proposal: dict[str, Any] | None
    refused: bool
    refusal_reason: str | None


@dataclass
class Citations:
    """인용 번호를 매긴다. 같은 청크는 한 번만 센다."""

    order: list[Chunk] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)

    def add(self, chunk: Chunk) -> int:
        if chunk.chunk_id not in self.index:
            self.order.append(chunk)
            self.index[chunk.chunk_id] = len(self.order)
        return self.index[chunk.chunk_id]

    def dump(self) -> list[dict[str, Any]]:
        return [chunk.citation(i + 1) for i, chunk in enumerate(self.order)]

    def __len__(self) -> int:
        return len(self.order)


@dataclass
class Read:
    """한 사람 몫으로 읽어 온 것."""

    profile: RunProfile
    factor: str
    band: str
    percentile: int | None
    rows: list[dict[str, Any]]
    chunks: list[Chunk]
    notices: list[str]


def target_factor(profile: RunProfile) -> tuple[str, str, int | None, list[dict[str, Any]]]:
    """가장 낮은 요인과 그 band·백분위. 측정이 없으면 빈 채로 돌아온다."""
    rows, _ = factor_rows(profile.profile())
    graded = [row for row in rows if row["percentile"] is not None]
    if not graded:
        return "", "", None, rows
    low = min(graded, key=lambda row: int(row["percentile"]))
    return str(low["factor"]), str(low["band"]), int(low["percentile"]), rows


def _prescriptions(profile: RunProfile, factor: str, band: str) -> tuple[list[Chunk], list[str]]:
    """처방 근거를 찾는다. 좁게 찾고, 비면 한 칸씩 넓힌다.

    국민체력100 처방은 나이 한 살 단위로 나뉘어 있고 모든 나이가 차 있지는 않다.
    여덟 살 자리가 비었다고 빈손으로 돌려보내지 않는다 — 같은 연령대의 가까운
    나이를 쓰고, 넓혔다고 말한다.
    """
    query = " ".join(
        part for part in [profile.age_group, f"{profile.age}세", factor, "운동처방"] if part
    )
    result = search(query, k=80, sources=("prescription",))
    parsed = []
    for hit in result.hits:
        matched = _CHUNK_ID.match(hit.chunk.chunk_id)
        if matched:
            parsed.append((hit.chunk, hit.score, matched))

    wanted = _BAND_GRADE.get(band, "")

    def pick(test) -> list[Chunk]:
        rows = [(c, s) for c, s, f in parsed if test(f)]
        rows.sort(key=lambda pair: (pair[0].grade != wanted, -pair[1]))
        return [chunk for chunk, _ in rows]

    same_sex = lambda f: f["sex"] == profile.sex  # noqa: E731
    same_factor = lambda f: not factor or f["factor"] == factor  # noqa: E731

    ladder = (
        (
            "",
            lambda f: int(f["age"]) == profile.age and same_sex(f) and same_factor(f),
        ),
        (
            f"만 {profile.age}세 처방 자료가 없어 같은 {profile.age_group} 자료를 썼습니다.",
            lambda f: f["age_group"] == profile.age_group and same_sex(f) and same_factor(f),
        ),
        (
            f"{profile.age_group} 자료가 없어 같은 성별의 다른 연령대 자료를 썼습니다.",
            lambda f: same_sex(f) and same_factor(f),
        ),
        (
            "요인을 좁히지 않고 그 연령대에 흔한 운동으로 골랐습니다.",
            lambda f: f["age_group"] == profile.age_group,
        ),
        ("연령대를 가리지 않고 두루 쓰이는 운동으로 골랐습니다.", lambda f: True),
    )

    for notice, test in ladder:
        chunks = pick(test)
        if chunks:
            return chunks[:6], [notice] if notice else []
    return [], []


def read_profile(profile: RunProfile) -> Read:
    factor, band, percentile, rows = target_factor(profile)
    chunks, notices = _prescriptions(profile, factor, band)
    return Read(profile, factor, band, percentile, rows, chunks, notices)


def _brief(read: Read) -> dict[str, Any]:
    measured = [
        {
            "요인": row["factor"],
            "백분위": row["percentile"],
            "상태": words.BAND_COPY.get(str(row["band"]), ""),
        }
        for row in read.rows
        if row["percentile"] is not None
    ]
    return {
        "ref": read.profile.ref,
        "역할": read.profile.role,
        "나이": read.profile.age,
        "나이단위": read.profile.age_unit,
        "성별": read.profile.sex,
        "연령대": read.profile.age_group,
        "대상_체력요인": read.factor or None,
        "측정": measured,
    }


def _rule_copy(read: Read, constraints: Constraints, slot: Slot) -> dict[str, str]:
    child = words.FOCUS_COPY.get(read.factor, "이번 주도 몸을 움직여 볼까요")
    if slot.weekly:
        child = "이번 주에 한 번은 다 같이 길게 움직여 볼까요"
    where = (
        f"주에 한 번 {slot.minutes}분"
        if slot.weekly
        else f"주 {constraints.days_per_week}회 {slot.minutes}분"
    )
    if read.factor and read.band:
        parent = f"{words.factor_copy(read.factor, read.band)}. {where}이면 충분합니다"
    else:
        parent = f"{where}으로 짰습니다. 측정을 하면 요인을 짚어 드릴 수 있습니다"
    return {"child": child, "parent": parent}


def _sessions_from(
    clips: list[tuple[str, catalog.Clip]],
    fallback_factor: str,
    evidence_base: list[int],
    citations: Citations,
) -> list[dict[str, Any]]:
    sessions = []
    for order, (phase, clip) in enumerate(clips, start=1):
        evidence = list(evidence_base)
        chunk = catalog.citation_for(clip.video_id)
        if chunk:
            evidence.append(citations.add(chunk))
        sessions.append(
            {
                "day_offset": 0,
                "phase": phase,
                "order": order,
                "exercise_name": clip.title,
                "fitness_factor": clip.fitness_factor or fallback_factor,
                "duration_sec": clip.duration_sec,
                "video": clip.as_video(),
                "evidence": sorted(set(evidence)),
            }
        )
    return sessions


def _mission(
    slot: Slot,
    sessions: list[dict[str, Any]],
    title: str,
    copy: dict[str, str],
    reason: str,
    start_date: date,
    weeks: int,
) -> dict[str, Any]:
    # 한 회 길이는 요청한 시간이다. 화면에서 한 편을 여러 세트 반복해 채우므로
    # 영상 길이의 합과 다르다 — 합은 video_sec 으로 따로 낸다.
    seconds = sum(int(session["duration_sec"]) for session in sessions)
    if slot.weekly:
        # 주간은 그 주 안에 아무 때나 한다. 날짜를 박지 않는다.
        first, last = start_date, start_date + timedelta(days=7 * weeks - 1)
    else:
        first = last = start_date + timedelta(days=slot.offset)
    return {
        "kind": slot.kind,
        "title": title,
        "period": {"start_date": first.isoformat(), "end_date": last.isoformat()},
        "participants": slot.participants,
        "duration_min": slot.minutes,
        "video_sec": seconds,
        "sessions": sessions,
        "copy": copy,
        "reason": reason,
    }


def _by_llm(
    read: Read,
    pool: list[catalog.Clip],
    slots: list[Slot],
    constraints: Constraints,
    evidence_base: list[int],
    citations: Citations,
    start_date: date,
    weeks: int,
) -> list[dict[str, Any]] | None:
    ids = {f"c{index}": clip for index, clip in enumerate(pool)}
    payload = {
        "참여자": _brief(read),
        "조건": {
            "조용히": constraints.quiet,
            "좁은_공간": constraints.small_space,
            "도구_없이": constraints.no_props,
        },
        "자리": [
            {
                "day_offset": slot.offset,
                "종류": slot.kind,
                "분": slot.minutes,
                "단계별_편수": catalog.clip_counts(slot.minutes),
            }
            for slot in slots
        ],
        "근거": [
            {"번호": index, "내용": citations.order[index - 1].text[:400]}
            for index in evidence_base
        ],
        "클립": [
            {
                "id": key,
                "이름": clip.title,
                "단계": clip.phase,
                "요인": clip.fitness_factor or "",
                "초": clip.duration_sec,
                "연령대": clip.age_group,
                "조용": clip.quiet,
                "좁은공간": clip.home_ok,
                "도구": clip.needs_props,
            }
            for key, clip in ids.items()
        ],
        "요청": (
            "자리마다 한 회씩 짠다. day_offset 은 그 자리 값이다. "
            f"day_offset 이 {WEEKLY_SLOT} 인 자리는 주간 미션으로, 그 주 안에 한 번 "
            "길게 온 가족이 함께 한다 — 날짜를 정하지 않는다. "
            "자리의 단계별_편수만큼만 고른다 — 화면에서 한 편을 여러 세트 반복해 "
            "시간을 채우므로 영상 길이의 합을 분에 맞출 필요가 없다."
        ),
    }

    days = coach_llm.plan_week(payload)
    if not days:
        return None

    by_offset = {slot.offset: slot for slot in slots}
    missions = []
    # 한 주 안에서 같은 동작을 다시 내지 않는다. 프롬프트로도 시키지만 지켜지지
    # 않아 한 주가 같은 차림으로 채워진 적이 있다.
    week: set[str] = set()
    for day_plan in days:
        slot = by_offset.get(int(day_plan.get("day_offset", 0)))
        if slot is None:
            continue
        # 고른 것을 단계별로 모은다. 목록에 없는 id 는 여기서 빠진다.
        offered: dict[str, list[catalog.Clip]] = {phase: [] for phase in catalog.PHASES}
        for row in day_plan.get("clips") or []:
            clip = ids.get(str(row.get("id")))
            if clip is None:
                continue
            phase = str(row.get("phase") or clip.phase)
            if phase in offered and clip.title not in {c.title for c in offered[phase]}:
                offered[phase].append(clip)

        # 단계마다 정해진 수만큼만 받는다. 이번 주에 안 쓴 것을 앞세우되, 그것으로
        # 모자라면 쓴 것도 받는다 — 빈 단계로 두는 것보다 낫다.
        want = catalog.clip_counts(slot.minutes)
        chosen: list[tuple[str, catalog.Clip]] = []
        seen: set[str] = set()
        for phase in catalog.PHASES:
            candidates = [c for c in offered[phase] if c.title not in seen]
            fresh = [c for c in candidates if c.title not in week]
            for clip in (fresh + [c for c in candidates if c.title in week])[: want[phase]]:
                seen.add(clip.title)
                chosen.append((phase, clip))
        if not chosen:
            continue
        week |= seen
        sessions = _sessions_from(chosen, read.factor, evidence_base, citations)
        if slot.weekly:
            fallback = "이번 주 함께 하기"
        else:
            weekday = (start_date + timedelta(days=slot.offset)).weekday()
            fallback = f"{_WEEKDAYS[weekday]} 운동"
        missions.append(
            _mission(
                slot,
                sessions,
                str(day_plan.get("title") or fallback),
                {
                    "child": str(day_plan.get("child") or ""),
                    "parent": str(day_plan.get("parent") or ""),
                },
                str(day_plan.get("reason") or ""),
                start_date,
                weeks,
            )
        )
    return missions or None


def _by_rule(
    read: Read,
    slots: list[Slot],
    constraints: Constraints,
    evidence_base: list[int],
    citations: Citations,
    start_date: date,
    weeks: int,
) -> list[dict[str, Any]]:
    prescribed = catalog.prescribed_names(read.chunks[:4])
    reason = (
        f"또래 처방에 나온 동작을 앞세워 골랐습니다 [{evidence_base[0]}]." if evidence_base else ""
    )
    used: set[str] = set()
    missions = []
    for slot in slots:
        picked = catalog.routine(
            read.profile.age_group,
            slot.minutes,
            factor=read.factor,
            prescribed=prescribed,
            conditions=constraints.conditions(),
            exclude=used,
        )
        flat = [(phase, clip) for phase in catalog.PHASES for clip in picked[phase]]
        if not flat:
            continue
        used |= {clip.title for _, clip in flat}
        sessions = _sessions_from(flat, read.factor, evidence_base, citations)
        if slot.weekly:
            title = f"이번 주 함께 {read.factor or '전신'} 기르기"
        else:
            day = start_date + timedelta(days=slot.offset)
            title = f"{_WEEKDAYS[day.weekday()]} {read.factor or '전신'} 기르기"
        missions.append(
            _mission(
                slot,
                sessions,
                title,
                _rule_copy(read, constraints, slot),
                reason,
                start_date,
                weeks,
            )
        )
    return missions


def build(
    profiles: list[RunProfile],
    start_date: date,
    weeks: int,
    constraints: Constraints,
) -> Plan:
    citations = Citations()
    movers = [p for p in profiles if p.role != "응원"]
    cheerers = [{"ref": p.ref, "role": p.role} for p in profiles if p.role == "응원"]

    if not movers:
        return Plan(
            [
                Step(1, "assess", "ok", "편성 대상이 없습니다 — 전원 응원"),
                Step(2, "retrieve", "failed", "검색하지 않음"),
                Step(3, "compose", "failed", "움직일 사람이 없어 중단"),
                Step(4, "verify", "failed", "인용 0건"),
            ],
            None,
            True,
            "no_relevant_source",
        )

    reads = [read_profile(mover) for mover in movers]
    driver = next((r for r in reads if r.profile.role == "주행자"), reads[0])
    measured = [r for r in reads if r.percentile is not None]
    assess_summary = (
        f"{driver.factor} 백분위 {driver.percentile} · 대상 요인 = {driver.factor}"
        if driver.percentile is not None
        else f"연령대 {driver.profile.age_group} · 만 {driver.profile.age}세 · 측정값 없음"
    )
    if len(reads) > 1:
        assess_summary += f" · 편성 대상 {len(reads)}명(측정 {len(measured)}명)"
    steps = [Step(1, "assess", "ok", assess_summary)]

    notices: list[str] = []
    chunk_total = sum(len(r.chunks) for r in reads)
    for read in reads:
        notices.extend(read.notices)

    if chunk_total == 0:
        steps += [
            Step(2, "retrieve", "failed", "처방 청크 0건"),
            Step(3, "compose", "failed", "편성 근거가 없어 중단"),
            Step(4, "verify", "failed", "인용 0건"),
        ]
        return Plan(steps, None, True, "no_relevant_source")

    day_slots = _WEEK_SLOTS.get(constraints.days_per_week, _WEEK_SLOTS[3])
    # 일간은 주행자가 받는다. 동반자는 제 몫을 따로 받지 않고 주간에 함께한다 —
    # 「동반자」가 그런 뜻이다. 주행자가 없으면 첫 사람을 주행자로 본다.
    drivers = [r for r in reads if r.profile.role == "주행자"] or reads[:1]
    everyone = [{"ref": r.profile.ref, "role": r.profile.role} for r in reads] + cheerers

    missions: list[dict[str, Any]] = []
    by_llm = 0
    clip_total = 0

    for read in drivers:
        if not read.chunks:
            continue
        prescribed = catalog.prescribed_names(read.chunks[:4])
        pool, pool_notice = catalog.pool(
            read.profile.age_group,
            conditions=constraints.conditions(),
            factor=read.factor,
            prescribed=prescribed,
        )
        if pool_notice:
            notices.append(pool_notice)
        if not pool:
            continue
        evidence_base = [citations.add(chunk) for chunk in read.chunks[:2]]
        mine = [{"ref": read.profile.ref, "role": read.profile.role}] + cheerers
        slots = [
            Slot("일간", offset, constraints.minutes_per_session, mine) for offset in day_slots
        ]
        # 주간은 온 가족이 한 번 길게 한다. 주행자가 여럿이어도 한 건만 둔다.
        if constraints.weekly_minutes and read is drivers[0]:
            slots.append(Slot("주간", WEEKLY_SLOT, constraints.weekly_minutes, everyone))

        theirs = _by_llm(
            read, pool, slots, constraints, evidence_base, citations, start_date, weeks
        )
        if theirs:
            by_llm += len(theirs)
        else:
            theirs = _by_rule(read, slots, constraints, evidence_base, citations, start_date, weeks)
        missions.extend(theirs)
        clip_total += sum(len(mission["sessions"]) for mission in theirs)

    steps.append(
        Step(2, "retrieve", "ok", f"처방 청크 {chunk_total}건 · 클립 후보 {len(catalog.clips())}개")
    )

    if not missions or not citations:
        steps += [
            Step(3, "compose", "failed", "편성 근거가 없어 중단"),
            Step(4, "verify", "failed", "인용 0건"),
        ]
        return Plan(steps, None, True, "no_citation_generated")

    how = "코치가" if by_llm else "규칙으로"
    steps.append(
        Step(
            3,
            "compose",
            "ok" if by_llm else "partial",
            f"미션 {len(missions)}건 · 클립 {clip_total}개 · {how} 편성",
        )
    )

    proposal: dict[str, Any] = {
        "missions": missions,
        "citations": citations.dump(),
    }
    if notices:
        proposal["notices"] = sorted(set(notices))
    return Plan(steps, proposal, False, None)


__all__ = ["Constraints", "Plan", "RunProfile", "Step", "build", "read_profile", "target_factor"]
