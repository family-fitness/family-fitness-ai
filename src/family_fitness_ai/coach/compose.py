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
from family_fitness_ai.common.copy import with_topic
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
    #: 보호자가 키워 주고 싶은 역량. 있으면 측정으로 고른 가장 낮은 요인보다 앞선다.
    focus_factor: str | None = None
    #: 보호자도 같이 한다. 참여자는 그대로고, LLM 코치가 문구를 쓸 때만 본다.
    with_companion: bool = False
    #: 그 사람이 최근 14일 동안 미션으로 받은 영상 id(유튜브 id 또는 공단 파일 이름),
    #: 최근 것부터. 후보를 고를 때 뒤로 미룬다 — 다른 후보가 모자랄 때만 다시 쓴다.
    recent_video_ids: tuple[str, ...] = ()

    @property
    def recent(self) -> frozenset[str]:
        return frozenset(self.recent_video_ids)

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
    #: 대상 요인을 보호자가 골랐나. False 면 측정으로 고른 가장 낮은 요인이다.
    focused: bool = False


def target_factor(
    profile: RunProfile, focus: str | None = None
) -> tuple[str, str, int | None, list[dict[str, Any]]]:
    """대상 요인과 그 band·백분위.

    보호자가 키워 주고 싶은 역량(focus)이 있으면 그것이 대상이다. 그 요인을 쟀으면
    band·백분위도 함께 내고, 안 쟀으면 빈 채로 낸다. 없으면 측정에서 가장 낮은
    요인을 고르고, 측정도 없으면 전부 빈 채로 돌아온다.
    """
    rows, _ = factor_rows(profile.profile())
    graded = [row for row in rows if row["percentile"] is not None]
    if focus:
        mine = next((row for row in graded if row["factor"] == focus), None)
        if mine is None:
            return focus, "", None, rows
        return focus, str(mine["band"]), int(mine["percentile"]), rows
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
    wanted_factor = factor or "그 요인"

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
        # 요인을 먼저 넓히고 성별은 끝까지 지킨다. 여자아이 처방이 있는데 남자아이
        # 처방을 근거로 들지 않는다.
        (
            f"{wanted_factor} 처방 자료가 없어 같은 나이 처방 가운데 흔한 운동으로 골랐습니다.",
            lambda f: int(f["age"]) == profile.age and same_sex(f),
        ),
        (
            f"{wanted_factor} 처방 자료가 없어 같은 {profile.age_group} 처방 가운데 "
            "흔한 운동으로 골랐습니다.",
            lambda f: f["age_group"] == profile.age_group and same_sex(f),
        ),
        (
            "같은 성별 처방 자료가 없어 요인과 성별을 좁히지 않고 "
            "그 연령대에 흔한 운동으로 골랐습니다.",
            lambda f: f["age_group"] == profile.age_group,
        ),
        ("연령대를 가리지 않고 두루 쓰이는 운동으로 골랐습니다.", lambda f: True),
    )

    for notice, test in ladder:
        chunks = pick(test)
        if chunks:
            return chunks[:6], [notice] if notice else []
    return [], []


def read_profile(profile: RunProfile, focus: str | None = None) -> Read:
    factor, band, percentile, rows = target_factor(profile, focus)
    chunks, notices = _prescriptions(profile, factor, band)
    return Read(profile, factor, band, percentile, rows, chunks, notices, focused=bool(focus))


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
        # 키울 요인을 부르는 말. 코치는 이 말로 부르고, 측정의 「상태」는 수준을 말할
        # 때만 쓴다(PLAN_SYSTEM).
        "대상_요인을_고른_까닭": (
            words.focus_reason(read.focused) if read.focused or read.factor else None
        ),
        "측정": measured,
    }


def _rule_copy(read: Read, constraints: Constraints, slot: Slot, weeks: int) -> dict[str, str]:
    child = words.FOCUS_COPY.get(read.factor, "이번 주도 몸을 움직여 볼까요")
    if slot.weekly:
        child = "이번 주에 한 번은 다 같이 길게 움직여 볼까요"
        where = f"주에 한 번 {slot.minutes}분"
    elif constraints.days_per_week == 1 and weeks == 1:
        # 한 주에 한 번, 한 주만 — 편성이 하루뿐이다(BE 의 하루 편성). 「주 1회」라고
        # 쓰면 한 주 계획으로 읽힌다. 날짜가 오늘이 아닐 수 있어 「오늘」로 쓰지 않는다.
        where = f"하루 {slot.minutes}분"
    else:
        where = f"주 {constraints.days_per_week}회 {slot.minutes}분"
    # 키울 요인은 고른 까닭으로 부른다. 「유연성은 꾸준히 하고 있는 영역입니다」처럼
    # 구간 문구로 부르면 왜 그걸 하라는지 읽히지 않는다.
    if read.focused:
        parent = f"{words.GUARDIAN_FOCUS}인 {read.factor}에 맞춰 {where}으로 짰습니다"
    elif read.factor and read.band:
        parent = f"{with_topic(read.factor)} {words.GROW_NOW}입니다. {where}이면 충분합니다"
    else:
        parent = f"{where}으로 짰습니다. 측정을 하면 요인을 짚어 드릴 수 있습니다"
    return {"child": child, "parent": parent}


def _rule_title(read: Read, slot: Slot, start_date: date) -> str:
    if slot.weekly:
        return f"이번 주 함께 {read.factor or '전신'} 기르기"
    day = start_date + timedelta(days=slot.offset)
    return f"{_WEEKDAYS[day.weekday()]} {read.factor or '전신'} 기르기"


def _weekday(slot: Slot, start_date: date) -> str | None:
    """일간 자리의 요일. 주간은 날을 정하지 않아 None 이다."""
    if slot.weekly:
        return None
    return _WEEKDAYS[(start_date + timedelta(days=slot.offset)).weekday()]


def _rule_reason(evidence_base: list[int]) -> str:
    if not evidence_base:
        return ""
    return f"또래 처방에 나온 동작을 앞세워 골랐습니다 [{evidence_base[0]}]."


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


@dataclass
class Tally:
    """코치의 편성을 우리가 손본 만큼. 편성 단계 요약에 싣는다 — 숨기지 않는다."""

    #: 코치가 모자라게 골라 목록에서 채운 클립
    filled: int = 0
    #: 길이·금지 어휘·빠진 동작 때문에 규칙 문구로 바꾼 칸
    rewritten: int = 0


def _by_llm(
    read: Read,
    pool: list[catalog.Clip],
    slots: list[Slot],
    constraints: Constraints,
    evidence_base: list[int],
    citations: Citations,
    start_date: date,
    weeks: int,
    tally: Tally,
) -> list[dict[str, Any]] | None:
    ids = {f"c{index}": clip for index, clip in enumerate(pool)}
    recent = constraints.recent
    payload = {
        "참여자": _brief(read),
        "조건": {
            "조용히": constraints.quiet,
            "좁은_공간": constraints.small_space,
            "도구_없이": constraints.no_props,
            "보호자도_함께": constraints.with_companion,
        },
        "자리": [
            {
                "day_offset": slot.offset,
                # 코치가 요일을 짐작하지 않게 준다. day_offset 만 주었을 때 수요일 편성
                # 제목이 「유연성을 키우는 월요일」로 왔다. 주간은 날을 정하지 않아 비운다.
                "날짜": (
                    None if slot.weekly else (start_date + timedelta(days=slot.offset)).isoformat()
                ),
                "요일": _weekday(slot, start_date),
                "종류": slot.kind,
                "분": slot.minutes,
                "단계별_편수": catalog.clip_counts(slot.minutes),
                **(
                    {
                        "본운동_대상_요인_편수": catalog.focus_quota(
                            catalog.clip_counts(slot.minutes)["본운동"]
                        )
                    }
                    if read.focused
                    else {}
                ),
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
                "영상": clip.video_id,
                "단계": clip.phase,
                "요인": clip.fitness_factor or "",
                "초": clip.duration_sec,
                "연령대": clip.age_group,
                "조용": clip.quiet,
                "좁은공간": clip.home_ok,
                "도구": clip.needs_props,
                "최근": clip.video_id in recent,
            }
            for key, clip in ids.items()
        ],
        "요청": (
            "자리마다 한 회씩 짠다. day_offset 은 그 자리 값이다. "
            f"day_offset 이 {WEEKLY_SLOT} 인 자리는 주간 미션으로, 그 주 안에 한 번 "
            "길게 온 가족이 함께 한다 — 날짜를 정하지 않는다. "
            "자리의 단계별_편수만큼만 고른다 — 화면에서 한 편을 여러 세트 반복해 "
            "시간을 채우므로 영상 길이의 합을 분에 맞출 필요가 없다. "
            "최근 이 true 인 클립은 이 사람이 요즘 받은 영상이다 — 같은 단계·요인의 다른 "
            "클립이 모자랄 때만 고른다. 한 회 안에서는 영상 이 같은 클립을 둘 넘게 "
            "고르지 않는다 — 다른 영상이 모자랄 때만 같은 영상을 다시 쓴다."
            + (
                f" 대상 요인 {read.factor} 은 보호자가 키워 주고 싶은 역량이다 — 본운동의 "
                "4분의 3 이상(자리의 본운동_대상_요인_편수만큼)을 요인 이 "
                f"{read.factor} 인 클립으로 고른다. 그런 클립이 모자라면 최근 이 true "
                "이거나 같은 영상인 것이라도 이 요인을 먼저 고르고, 그래도 모자라면 다른 "
                "요인으로 채운다. 준비운동 · 정리운동은 이 비율과 상관없다."
                if read.focused
                else ""
            )
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
        # 하나도 못 건진 날은 코치의 편성이 아니다. 목록으로 다 채우지 않고 버린다.
        if not any(offered.values()):
            continue

        chosen = _fill(
            offered,
            pool,
            catalog.clip_counts(slot.minutes),
            week,
            tally,
            recent,
            start_date.isoformat(),
            read.factor if read.focused else "",
        )
        week |= {clip.title for _, clip in chosen}
        # 코치가 골랐는데 이 회에서 빠진 동작. 글에 그 이름이 남아 있으면 못 쓴다.
        picked = {clip.title for clips in offered.values() for clip in clips}
        dropped = picked - {clip.title for _, clip in chosen}
        text = _checked_text(
            {key: day_plan.get(key) for key in ("title", "child", "parent", "reason")},
            {
                "title": _rule_title(read, slot, start_date),
                **_rule_copy(read, constraints, slot, weeks),
                "reason": _rule_reason(evidence_base),
            },
            dropped,
            _weekday(slot, start_date),
            tally,
        )
        missions.append(
            _mission(
                slot,
                _sessions_from(chosen, read.factor, evidence_base, citations),
                text["title"],
                {"child": text["child"], "parent": text["parent"]},
                text["reason"],
                start_date,
                weeks,
            )
        )
    return missions or None


def _fill(
    offered: dict[str, list[catalog.Clip]],
    pool: list[catalog.Clip],
    want: dict[str, int],
    week: set[str],
    tally: Tally,
    recent: frozenset[str] = frozenset(),
    seed: str = "",
    focus: str = "",
) -> list[tuple[str, catalog.Clip]]:
    """단계마다 정한 편수를 채운다.

    코치가 고른 새 것 → 목록의 새 것 → 코치가 고른 헌 것 → 목록의 헌 것 차례다.
    헌 것은 이번 주에 쓴 동작이거나 최근(recent)에 받은 영상이다. 코치는 편수를
    모자라게 고르거나 이미 쓴 동작 · 최근 영상을 또 고른다 — 프롬프트로 시켜도
    그렇다. 목록은 순위대로 서 있어 앞에서부터 채운다. 목록에도 없으면 모자란
    채로 둔다. 조건을 몰래 풀지 않는다.

    한 회 안에서 한 영상이 여러 칸을 채우지 않게, 먼저 그 회에 아직 없는 영상으로만
    채우고 모자라면 그때 같은 영상의 다른 클립을 쓴다. 한 영상의 클립이 일곱 칸 중
    다섯 칸을 채운 적이 있다. 후보가 한 영상뿐이면(유아기) 그대로 다시 쓴다.

    한 단계의 후보가 모두 헌 것이면 코치가 고른 것을 앞세우지 않고 seed(편성
    시작일)로 섞는다. 유아기 정리운동 후보 일곱이 모두 한 영상에서 나와 늘 최근이라,
    코치가 날마다 고르는 같은 클립이 14일 중 13일 나왔다.

    focus(보호자가 키워 주고 싶은 역량)가 있으면 본운동은 먼저 4분의 3 칸
    (catalog.focus_quota)을 그 역량 클립으로 채운다. 코치가 고른 그 역량 클립 →
    목록의 그 역량 클립 차례고, 헌 것 · 한 회 안에서 이미 나온 영상이라도 그 역량이면
    다른 요인보다 먼저 쓴다(사용자 결정: 그 역량 → 새 영상). 그 역량이 모자라면
    남은 칸은 위의 차례대로 채운다. 코치가 그 역량이 아닌 본운동을 넷 골랐으면
    뒤의 것부터 밀려난다. 준비 · 정리운동은 focus 로 바뀌지 않는다.
    """
    chosen: list[tuple[str, catalog.Clip]] = []
    today: set[str] = set()
    videos: set[str] = set()
    # 코치가 그날 고른 동작은 목록에서 채울 때 건드리지 않는다. 늘리는 동작은 준비·정리
    # 두 단계에 다 있어서, 앞 단계를 채우다 코치가 뒤 단계에 둔 것을 먼저 가져간 적이 있다.
    reserved = {clip.title for clips in offered.values() for clip in clips}
    # 영상도 같다(reserved_videos). 목록에서 채우다 코치가 뒤 단계에 둔 영상의 다른
    # 클립을 먼저 넣으면, 뒤 단계에서 코치가 고른 것이 「이미 나온 영상」 이 되어 밀려난다.
    for index, phase in enumerate(catalog.PHASES):
        reserved_videos = {
            clip.video_id for later in catalog.PHASES[index + 1 :] for clip in offered[later]
        }
        picked = offered[phase]
        spare = [clip for clip in pool if clip.phase == phase and clip.title not in reserved]

        def worn(clip: catalog.Clip) -> bool:
            return clip.title in week or clip.video_id in recent

        order = (
            [clip for clip in picked if not worn(clip)]
            + [clip for clip in spare if not worn(clip)]
            + [clip for clip in picked if worn(clip)]
            + [clip for clip in spare if worn(clip)]
        )
        if seed and order and all(worn(clip) for clip in order):
            order.sort(key=lambda clip: (clip.title in week, catalog.shuffle_key(clip, seed)))
        rounds = [(order, want[phase])]
        if focus and phase == "본운동":
            mine = [clip for clip in order if clip.fitness_factor == focus]
            # 그 역량 안에서는 코치가 고른 새 것 → 목록의 새 것 → 헌 것 차례를 지킨다.
            rounds.insert(0, (mine, min(want[phase], catalog.focus_quota(want[phase]))))
        count = 0
        for candidates, limit in rounds:
            for other_videos_only in (True, False):
                for clip in candidates:
                    if count >= limit:
                        break
                    if clip.title in today:
                        continue
                    if other_videos_only and (
                        clip.video_id in videos
                        or (clip not in picked and clip.video_id in reserved_videos)
                    ):
                        continue
                    today.add(clip.title)
                    videos.add(clip.video_id)
                    chosen.append((phase, clip))
                    count += 1
                    if clip not in picked:
                        tally.filled += 1
    return chosen


def _checked_text(
    written: dict[str, Any],
    fallback: dict[str, str],
    dropped: set[str],
    weekday: str | None,
    tally: Tally,
) -> dict[str, str]:
    """코치가 쓴 글을 칸마다 잰다. 어긋난 칸만 규칙 문구로 바꾼다.

    길이(title 16 · child 45 · parent 70)와 금지 어휘, 이 회에서 빠진 동작의
    이름, 그 자리와 다른 요일(주간이면 어느 요일이든), 그리고 제목에 요인 이름
    대신 들어온 이유 문구(「지금 키우기 좋은 영역」)를 본다. 프롬프트로
    시키지만 지켜지지 않았다 — 주간 parent 가 92자로 나간 적이 있다. 반쯤 고쳐 쓰지
    않는다. 그 칸을 통째로 바꾼다.
    """
    out: dict[str, str] = {}
    for key, rule in fallback.items():
        value = written.get(key)
        limit = coach_llm.LIMITS.get(key)
        fits = (
            isinstance(value, str)
            and bool(value.strip())
            and (limit is None or len(value) <= limit)
            and not words.banned_words_in(value)
            and not any(name in value for name in dropped)
            and not any(day in value for day in _WEEKDAYS if day != weekday)
            and not (key == "title" and coach_llm.names_the_reason(value))
        )
        if fits:
            out[key] = str(value)
        else:
            out[key] = rule
            tally.rewritten += 1
    return out


def _in_order(missions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """일간은 날짜순, 주간은 맨 뒤. 같은 날이면 주행자 차례를 지킨다.

    코치가 답한 차례대로 두면 주간이 맨 앞에 오고, 주행자가 둘이면 날짜가 두 번
    돈다. 정렬은 안정적이라 같은 날의 주행자 차례는 그대로다.
    """
    return sorted(missions, key=lambda m: (m["kind"] == "주간", m["period"]["start_date"]))


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
    reason = _rule_reason(evidence_base)
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
            level=catalog.level_of(read.percentile),
            # BE 는 하루씩 부른다. 날짜로 섞고 최근에 받은 영상을 미뤄 날마다 같은
            # 영상이 나오지 않게 한다.
            seed=start_date.isoformat(),
            recent=constraints.recent,
            # 보호자가 키워 주고 싶은 역량이면 본운동 4분의 3 을 그 역량으로 채운다.
            focus=read.factor if read.focused else "",
        )
        flat = [(phase, clip) for phase in catalog.PHASES for clip in picked[phase]]
        if not flat:
            continue
        used |= {clip.title for _, clip in flat}
        sessions = _sessions_from(flat, read.factor, evidence_base, citations)
        missions.append(
            _mission(
                slot,
                sessions,
                _rule_title(read, slot, start_date),
                _rule_copy(read, constraints, slot, weeks),
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

    # 보호자가 키워 주고 싶은 역량은 일간을 받는 사람(주행자) 몫이다. 동반자는
    # 제 몫을 따로 받지 않으니 측정으로 고른 요인을 그대로 둔다.
    steering = {id(p) for p in movers if p.role == "주행자"} or {id(movers[0])}
    reads = [
        read_profile(mover, constraints.focus_factor if id(mover) in steering else None)
        for mover in movers
    ]
    driver = next((r for r in reads if r.profile.role == "주행자"), reads[0])
    measured = [r for r in reads if r.percentile is not None]
    # 잰 항목이 있는데 백분위가 비면 또래 기준이 없는 것이다(7–10세는 늘 그렇다).
    # 「측정값 없음」은 잰 항목이 정말 없을 때만 쓴다.
    no_peer = "또래 비교 기준 없음"
    if driver.focused:
        focus_measured = any(row["factor"] == driver.factor for row in driver.rows)
        scored = (
            f"{driver.factor} 백분위 {driver.percentile}"
            if driver.percentile is not None
            else f"{driver.factor} {no_peer if focus_measured else '측정값 없음'}"
        )
        assess_summary = f"{scored} · 대상 요인 = {driver.factor}({words.GUARDIAN_FOCUS})"
    elif driver.percentile is not None:
        assess_summary = (
            f"{driver.factor} 백분위 {driver.percentile} · "
            f"대상 요인 = {driver.factor}({words.GROW_NOW})"
        )
    else:
        assess_summary = (
            f"연령대 {driver.profile.age_group} · 만 {driver.profile.age}세 · "
            f"{no_peer if driver.rows else '측정값 없음'}"
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
    tally = Tally()

    for read in drivers:
        if not read.chunks:
            continue
        prescribed = catalog.prescribed_names(read.chunks[:4])
        pool, pool_notice = catalog.pool(
            read.profile.age_group,
            conditions=constraints.conditions(),
            factor=read.factor,
            prescribed=prescribed,
            # 공단 영상에는 알맞은 체력수준이 적혀 있다. 대상 요인의 백분위로 맞춘다.
            level=catalog.level_of(read.percentile),
            seed=start_date.isoformat(),
            recent=constraints.recent,
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
            read, pool, slots, constraints, evidence_base, citations, start_date, weeks, tally
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
    summary = f"미션 {len(missions)}건 · 클립 {clip_total}개 · {how} 편성"
    # 코치의 편성을 손봤으면 그만큼 말한다.
    if tally.filled:
        summary += f" · 목록에서 {tally.filled}편 채움"
    if tally.rewritten:
        summary += f" · 문구 {tally.rewritten}칸 규칙으로"
    steps.append(Step(3, "compose", "ok" if by_llm else "partial", summary))

    proposal: dict[str, Any] = {
        "missions": _in_order(missions),
        "citations": citations.dump(),
    }
    if notices:
        proposal["notices"] = sorted(set(notices))
    return Plan(steps, proposal, False, None)


__all__ = ["Constraints", "Plan", "RunProfile", "Step", "build", "read_profile", "target_factor"]
