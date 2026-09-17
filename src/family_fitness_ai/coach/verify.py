"""내보내기 직전의 마지막 관문.

넷을 본다 — 인용이 하나라도 있나, 본문의 [n] 이 인용 범위 안인가, 금지 어휘가
없나, 세션의 연령대가 프로필과 맞나. **부분 통과는 없다.** 하나라도 걸리면
제안을 내보내지 않는다.

여기서 보는 것은 화면에 나가는 글자뿐이다. chunk_id 같은 식별자는 코드값이라
금지 어휘 검사에서 뺀다.
"""

from __future__ import annotations

import re
from typing import Any

from family_fitness_ai.common import copy as words
from family_fitness_ai.video import catalog

_MARK = re.compile(r"\[(\d+)\]")


def check_proposal(proposal: dict[str, Any], age_groups: set[str]) -> list[str]:
    """어긋난 것들. 빈 목록이면 통과다."""
    problems: list[str] = []
    citations = list(proposal.get("citations") or [])
    missions = list(proposal.get("missions") or [])

    if not citations:
        problems.append("인용 0건")
    allowed = {int(citation["index"]) for citation in citations}

    for mission in missions:
        texts = [
            str(mission.get("title", "")),
            str(mission.get("reason", "")),
            *map(str, dict(mission.get("copy") or {}).values()),
        ]
        for text in texts:
            found = words.banned_words_in(text)
            if found:
                problems.append(f"금지 어휘 {'·'.join(found)}")
            for mark in _MARK.findall(text):
                if int(mark) not in allowed:
                    problems.append(f"인용 범위 밖 [{mark}]")

        for session in mission.get("sessions") or []:
            for index in session.get("evidence") or []:
                if int(index) not in allowed:
                    problems.append(f"인용 범위 밖 근거 {index}")
            video = session.get("video")
            if not video:
                continue
            # 연령대가 다른 영상은 막지 않는다. 라벨이 붙은 영상이 많지 않아
            # 또래만 고집하면 아무것도 못 준다 — 섞였다는 사실은 notices 로
            # 알리고, 쓸지는 화면 저쪽에서 정한다.
            if not catalog.citation_for(str(video["video_id"])):
                problems.append(f"출처 없는 영상 {video['video_id']}")

    return sorted(set(problems))


def check_answer(answer: str, citations: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    if not citations:
        problems.append("인용 0건")
    allowed = {int(citation["index"]) for citation in citations}
    marks = {int(mark) for mark in _MARK.findall(answer)}
    if not marks:
        problems.append("본문에 근거 번호가 없다")
    for mark in marks - allowed:
        problems.append(f"인용 범위 밖 [{mark}]")
    found = words.banned_words_in(answer)
    if found:
        problems.append(f"금지 어휘 {'·'.join(found)}")
    return sorted(set(problems))
