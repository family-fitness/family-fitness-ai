"""화면에 나가는 문구.

band·factor 같은 코드값은 화면에 나가지 않는다. 표시 문구는 전부 여기서 나온다.
"부족"·"미달"·"하위"를 쓰지 않는다 — 아이가 읽는 화면이다.
"""

from __future__ import annotations

DISCLAIMER = (
    "국민체력100 측정 데이터를 바탕으로 한 참고 정보입니다. "
    "질병의 진단·치료를 위한 것이 아니며, 건강에 관한 판단은 전문가와 상담하세요."
)

TRAJECTORY_NOTICE = "집단 분포를 바탕으로 한 참고 범위입니다. 개인의 변화를 나타내지 않습니다."

#: band 코드 → 부모가 읽는 말.
BAND_COPY = {
    "strength": "잘하고 있는 영역",
    "steady": "꾸준히 하고 있는 영역",
    "growth": "지금 키우기 좋은 영역",
}

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
    """받침을 보고 은/는을 붙인다. 「유연성은」 · 「자세는」."""
    if not noun:
        return noun
    last = ord(noun[-1])
    if 0xAC00 <= last <= 0xD7A3:
        return noun + ("은" if (last - 0xAC00) % 28 else "는")
    return noun + "은(는)"


def factor_copy(factor: str, band: str) -> str:
    """「유연성은 지금 키우기 좋은 영역입니다」."""
    return f"{with_topic(factor)} {BAND_COPY[band]}입니다"


def banned_words_in(text: str) -> list[str]:
    return [w for w in BANNED_WORDS if w in text]
