"""화면에 나가는 문구.

band·factor 같은 코드값은 화면에 나가지 않는다. 표시 문구는 전부 여기서 나온다.
"부족", "미달", "하위"를 쓰지 않는다. 아이가 읽는 화면이다.

가운데 점(·)과 긴 대시(—, –)도 쓰지 않는다. AI 가 쓴 티가 난다(사용자 결정). 여기
적는 문구에 넣지 않고, 코치(LLM)나 원자료에서 들어온 글은 plain() 으로 바꿔 내보낸다.
"""

from __future__ import annotations

import re

DISCLAIMER = (
    "국민체력100 측정 결과로 만든 참고 정보입니다. "
    "병을 진단하거나 치료하려고 쓰는 정보가 아니니, 건강이 걱정되면 전문가와 상담하세요."
)

TRAJECTORY_NOTICE = "집단 분포를 바탕으로 한 참고 범위입니다. 개인의 변화를 나타내지 않습니다."

#: band 코드 → 보호자가 읽는 말. 그 요인의 **수준**을 말할 때만 쓴다.
BAND_COPY = {
    "strength": "잘하고 있는 영역",
    "steady": "꾸준히 하고 있는 영역",
    "growth": "지금 키우기 좋은 영역",
}

#: 이번 편성이 키울 요인을 부르는 말. 고른 까닭으로 부른다 — 구간 문구(BAND_COPY)로
#: 부르면 「꾸준히 하고 있는 영역」을 왜 하라는지 읽히지 않는다. 앞의 것은 FE 결과
#: 화면과 같은 말이다.
GROW_NOW = "지금 키우기 좋은 영역"
GUARDIAN_FOCUS = "보호자가 키워 주고 싶은 역량"


def focus_reason(focused: bool) -> str:
    """키울 요인을 고른 까닭. 보호자가 골랐으면 그 말, 아니면 가장 낮은 요인이다."""
    return GUARDIAN_FOCUS if focused else GROW_NOW


#: verify 가 반환 직전에 거르는 말. 하나라도 있으면 통과하지 않는다.
BANNED_WORDS = (
    "부족",
    "미달",
    "하위",
    "열등",
    "비만",
    "저체중",
    "낙제",
    "뒤떨어",
    "문제가 있습니다",
)

#: 요인별로 아이에게 건네는 권유. 처방이 비어도 이 말은 근거 없이 서지 않는다 —
#: 요인을 지목한 뒤에만 쓴다.
FOCUS_COPY = {
    "심폐지구력": "이번 주는 숨이 조금 차오를 때까지 움직여 볼까요",
    "근력": "이번 주는 힘껏 밀고 당기는 동작을 해볼까요",
    "근지구력": "이번 주는 같은 동작을 천천히 여러 번 해볼까요",
    "유연성": "이번 주는 몸을 길게 늘이는 동작을 해볼까요",
    "민첩성": "이번 주는 방향을 빠르게 바꾸는 동작을 해볼까요",
    "순발력": "이번 주는 힘껏 뛰어오르는 동작을 해볼까요",
    "협응력": "이번 주는 눈과 손을 같이 쓰는 동작을 해볼까요",
    "평형성": "이번 주는 한 발로 버티는 동작을 해볼까요",
}


def band_of(percentile: int | None) -> str | None:
    """백분위를 band 로. 미측정이면 None 이다."""
    if percentile is None:
        return None
    if percentile >= 75:
        return "strength"
    if percentile >= 25:
        return "steady"
    return "growth"


def with_topic(noun: str) -> str:
    """받침을 보고 은/는을 붙인다. 「유연성은」, 「자세는」."""
    if not noun:
        return noun
    last = ord(noun[-1])
    if 0xAC00 <= last <= 0xD7A3:
        return noun + ("은" if (last - 0xAC00) % 28 else "는")
    return noun + "은(는)"


def with_subject(noun: str) -> str:
    """받침을 보고 이/가를 붙인다. 「유연성이」 · 「자세가」."""
    if not noun:
        return noun
    last = ord(noun[-1])
    if 0xAC00 <= last <= 0xD7A3:
        return noun + ("이" if (last - 0xAC00) % 28 else "가")
    return noun + "이(가)"


#: 「2등급」 「3 등급」. 화면에 나오는 등급은 국민체력100 등급 카드(한 사람에 하나)
#: 뿐이다. 처방표는 요인마다 등급 칸으로 나뉘어 있어, 코치가 그 칸 이름을 옮겨
#: 「심폐지구력 2등급」 처럼 요인별 등급을 쓴 적이 있다.
_GRADE = re.compile(r"\d\s*등급")


def grades_in(text: str) -> set[str]:
    """글에 나오는 등급 말. 「2 등급」도 「2등급」으로 센다."""
    return {re.sub(r"\s", "", found) for found in _GRADE.findall(text)}


def factor_copy(factor: str, band: str) -> str:
    """「유연성은 지금 키우기 좋은 영역입니다」."""
    return f"{with_topic(factor)} {BAND_COPY[band]}입니다"


def banned_words_in(text: str) -> list[str]:
    return [w for w in BANNED_WORDS if w in text]


#: 가운데 점으로 보이는 글자. 한글 자판의 「ㆍ」 와 일본식 「・」 도 화면에서는 같다.
_DOTS = "·ㆍ・‧∙"
#: 긴 대시로 보이는 글자. 붙임표(-)는 동작 이름(「T-W-Y-A」)에 쓰여 건드리지 않는다.
_DASHES = "—–―‒"
#: 「7–10세」 「3 — 5회」. 숫자 사이의 대시는 범위라 물결로 바꾼다.
_RANGE = re.compile(rf"(\d)\s*[{_DASHES}]\s*(?=\d)")
_MARK = re.compile(rf"\s*[{_DOTS}{_DASHES}]+\s*")
#: 바꾸고 나서 쉼표가 문장부호나 괄호 앞에 붙거나 글 끝에 남은 것.
_LOOSE_COMMA = re.compile(r",\s*(?=[,.!?)\]」』]|$)")
_LEADING_COMMA = re.compile(r"(^|[(\[「『])\s*,\s*")
#: 근거 번호([1]) 앞의 쉼표. 「늘여 봐요 — [1].」 은 「늘여 봐요 [1].」 이다.
_COMMA_BEFORE_MARK = re.compile(r"\s*,\s*(?=\[\d+\])")


def plain(text: str) -> str:
    """가운데 점과 긴 대시를 걷어 낸 글.

    숫자 사이의 대시는 범위라 물결(7~10세)로, 나머지 점과 대시는 쉼표로 바꾼다.
    「국민체력100 운동처방동영상 · 걷기」 → 「국민체력100 운동처방동영상, 걷기」,
    「편성은 끝났다 — 너는」 → 「편성은 끝났다, 너는」. 글 앞뒤나 문장부호 앞에 남는
    쉼표는 뺀다.
    """
    if not text or not any(mark in text for mark in _DOTS + _DASHES):
        return text
    text = _RANGE.sub(r"\1~", text)
    text = _MARK.sub(", ", text)
    text = _COMMA_BEFORE_MARK.sub(" ", text)
    text = _LOOSE_COMMA.sub("", text)
    text = _LEADING_COMMA.sub(r"\1", text)
    return text.strip()
