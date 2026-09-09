"""화면에 나가는 문구 (docs/dev/AI-4 §3.3).

**문구는 규칙으로 만든다. LLM이 아니다.** `docs/01` §3.1 이 LLM 호출을 `compose`
한 곳으로 묶었고, `assessment` 는 그 그래프를 타지 않는다.

한 모듈에 모으는 이유는 나중에 `verify` 가 같은 표를 봐야 하기 때문이다
(`docs/03` §5.4). 문구가 코드에 흩어지면 금지 어휘 검사가 닿지 못하는 자리가 생긴다.
"""

from __future__ import annotations

from .types import AgeGroup, Band, FitnessFactor

# docs/03 §2.6 — "부족"·"미달"·"하위"를 쓰지 않는다. 순위가 아니라 활동으로 말한다.
FORBIDDEN_WORDS = ("부족", "미달", "하위", "열등", "낙제", "미흡")

# docs/03 §3.5 의 표가 정본이다.
BAND_PHRASE: dict[Band, str] = {
    "strength": "잘하고 있는 영역",
    "steady": "꾸준히 하고 있는 영역",
    "growth": "지금 키우기 좋은 영역",
}

# 아이 화면 문구. 요인 하나에 하나이고, 순위·비교를 담지 않는다 (docs/03 §3.3).
CHILD_FOCUS: dict[FitnessFactor, str] = {
    "심폐지구력": "이번 주는 숨이 차오를 만큼 신나게 움직여볼까요",
    "근력": "이번 주는 몸을 밀고 당기는 동작을 해볼까요",
    "근지구력": "이번 주는 같은 동작을 여러 번 반복해볼까요",
    "유연성": "이번 주는 몸을 길게 늘이는 동작을 해볼까요",
    "민첩성": "이번 주는 방향을 바꾸며 재빠르게 움직여볼까요",
    "순발력": "이번 주는 힘껏 뛰어오르는 동작을 해볼까요",
    "협응력": "이번 주는 손과 눈을 함께 쓰는 놀이를 해볼까요",
    "평형성": "이번 주는 한 발로 서서 균형을 잡아볼까요",
}

# 측정값이 없을 때(`L0`·`L1`)의 연령대 고정 제안.
#
# **점수가 없으므로 요인을 고르는 근거도 없다** (docs/dev/AI-4 §3.1). 이 표는
# 근거가 아니라 자리를 채우는 값이고, 코퍼스가 서면 안전지침 청크가 근거가 된다
# (docs/dev/AI-4 §6 ②). 바꿀 때 코드를 뒤질 필요가 없도록 여기 모아 둔다.
AGE_GROUP_FOCUS: dict[AgeGroup, FitnessFactor] = {
    "유아기": "협응력",
    "유소년": "순발력",
    "청소년": "심폐지구력",
    "성인": "심폐지구력",
    "어르신": "평형성",
}


def child_focus(factor: FitnessFactor) -> str:
    return CHILD_FOCUS[factor]


def _topic_particle(word: str) -> str:
    """받침이 있으면 `은`, 없으면 `는`.

    지금 요인 여덟은 전부 받침으로 끝나 늘 `은` 이지만, 요인이 늘었을 때 조사가
    어긋나는 것을 코드가 아니라 글자로 판단하게 둔다.
    """
    last = word[-1]
    if "가" <= last <= "힣":
        return "은" if (ord(last) - 0xAC00) % 28 else "는"
    return "은(는)"


def parent_line(factor: str, band: Band) -> str:
    """부모 화면 한 줄. `docs/03` §3.2 의 `copy.strength`·`copy.focus` 가 이 모양이다."""
    return f"{factor}{_topic_particle(factor)} {BAND_PHRASE[band]}입니다"


def contains_forbidden(text: str) -> list[str]:
    """금지 어휘를 찾아 돌려준다. 검사와 `verify` 가 같은 목록을 본다."""
    return [w for w in FORBIDDEN_WORDS if w in text]
