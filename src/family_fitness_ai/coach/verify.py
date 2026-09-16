"""반환 직전 검사 (docs/03 §5.4).

넷을 본다 — **인용 1개 이상 · 본문 `[n]` 이 인용 범위 안 · 문구에 금지 어휘 없음 ·
세션 연령대가 프로필과 일치.** **부분 통과는 없다.** 하나라도 걸리면 거부다.

검사가 제안을 **고치지 않는다.** 고쳐서 통과시키면 검사가 아니라 세탁이다 —
걸린 것은 거부로 내보내고 사유를 남긴다.

연령 검사는 `plan` 이 이미 본 것을 **독립된 자료로 다시 본다** — 편성은 후보에
붙일 때 보고, 여기서는 완성된 제안과 영상 표를 맞대어 본다. 같은 함수
(`select.video_allowed`)를 쓰는 이유는 규칙이 두 벌이 되면 갈라지기 때문이다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from ..api.coach_schemas import Mission, Proposal
from ..common.copy import contains_forbidden
from ..mission.select import video_allowed

# 본문에 박힌 인용 번호. `coach/messages` 의 답변에도 같은 것을 쓴다.
CITE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Failure:
    check: str
    detail: str


def cite_numbers(text: str) -> list[int]:
    return [int(m) for m in CITE.findall(text)]


def check_proposal(
    proposal: Proposal,
    # ref → (연령대, 나이, 나이 단위)
    profiles: Mapping[str, tuple[str, int, str]],
    # video_id → 그 영상의 연령대. 모르는 영상은 **검사에서 통과시키지 않는다**
    video_groups: Mapping[str, str],
) -> list[Failure]:
    """걸린 것 전부. 빈 목록이면 통과다."""
    out: list[Failure] = []

    if not proposal.citations:
        return [Failure("citation", "인용이 0건이다")]

    valid = {c.index for c in proposal.citations}

    for position, mission in enumerate(proposal.missions):
        where = f"missions[{position}]"

        for session in mission.sessions:
            if outside := sorted(set(session.evidence) - valid):
                out.append(Failure("evidence", f"{where} 의 근거 {outside} 가 인용 범위 밖이다"))

        for name, text in (
            ("title", mission.title),
            ("copy.child", mission.text.child),
            ("copy.parent", mission.text.parent),
        ):
            if found := contains_forbidden(text):
                out.append(Failure("forbidden", f"{where}.{name} 에 금지 어휘 {found}"))
            if outside := sorted(set(cite_numbers(text)) - valid):
                out.append(Failure("evidence", f"{where}.{name} 의 {outside} 가 인용 범위 밖이다"))

        out += _age_failures(mission, where, profiles, video_groups)
    return out


def _age_failures(
    mission: Mission,
    where: str,
    profiles: Mapping[str, tuple[str, int, str]],
    video_groups: Mapping[str, str],
) -> list[Failure]:
    """이 미션의 영상이 **참여자 전원에게** 맞나 (docs/02 §2.3).

    한 명에게라도 맞지 않으면 걸린다 — 미션은 참여자가 함께 보는 카드이고,
    영상 하나가 그 카드에 붙는다 (`ProposalConverter` 가 미션당 첫 영상 하나만 살린다).
    """
    out: list[Failure] = []
    for session in mission.sessions:
        if session.video is None:
            continue
        video_id = session.video.video_id
        group = video_groups.get(video_id)
        if group is None:
            out.append(
                Failure("age", f"{where} 의 영상 {video_id} 가 영상 표에 없다 — 연령을 알 수 없다")
            )
            continue
        for participant in mission.participants:
            known = profiles.get(participant.ref)
            if known is None:
                out.append(Failure("age", f"{where} 의 참여자 {participant.ref} 가 요청에 없다"))
                continue
            age_group, age, age_unit = known
            if not video_allowed(age_group, age, age_unit, group):
                out.append(
                    Failure(
                        "age",
                        f"{where} 의 영상 {video_id}({group}) 이 "
                        f"{participant.ref}({age_group} {age}{age_unit}) 에게 맞지 않는다",
                    )
                )
    return out
