"""계약의 공통 타입 (docs/03 §2.4).

문자열 리터럴을 라우터에 흩지 않는다. docs/02 §2.1 이 경고한 대로 `유아기` 만
'기'가 붙고 나머지는 붙지 않는다 — 패턴으로 추론하지 말고 상수로 고정한다.

값이 계약에 속하므로 정본은 여기다. `stats` 와 `ingest` 가 가져다 쓴다.
"""

from __future__ import annotations

from typing import Literal, get_args

AgeGroup = Literal["유아기", "유소년", "청소년", "성인", "어르신"]
Sex = Literal["M", "F"]
AgeUnit = Literal["세", "개월"]
InputLevel = Literal["L0", "L1", "L2"]
Band = Literal["strength", "steady", "growth"]
Role = Literal["주행자", "동반자", "응원"]
FitnessFactor = Literal[
    "심폐지구력", "근력", "근지구력", "유연성", "민첩성", "순발력", "협응력", "평형성"
]

AGE_GROUPS: tuple[AgeGroup, ...] = get_args(AgeGroup)
SEXES: tuple[Sex, ...] = get_args(Sex)
AGE_UNITS: tuple[AgeUnit, ...] = get_args(AgeUnit)
INPUT_LEVELS: tuple[InputLevel, ...] = get_args(InputLevel)
BANDS: tuple[Band, ...] = get_args(Band)
ROLES: tuple[Role, ...] = get_args(Role)
FITNESS_FACTORS: tuple[FitnessFactor, ...] = get_args(FitnessFactor)

# 아이 프로필. 연령 안전 규칙은 이 셋에 적용된다 (docs/03 §2.4).
CHILD_AGE_GROUPS: tuple[AgeGroup, ...] = ("유아기", "유소년", "청소년")

# 점수를 내는 연령대. 어르신은 기준항목이 미정이라 제외한다 (docs/02 §6 ⑦).
SCORED_AGE_GROUPS: tuple[AgeGroup, ...] = ("유아기", "유소년", "청소년", "성인")

# 혈압. 값이 있어도 전송하지 않는다 — 전송되면 400 이다 (docs/03 §9).
BLOCKED_ITEM_CODES: frozenset[str] = frozenset({"005", "006"})

# 유아기만 개월이다 (docs/02 §2.4).
MONTH_AGE_GROUP: AgeGroup = "유아기"


def age_unit_of(age_group: AgeGroup) -> AgeUnit:
    return "개월" if age_group == MONTH_AGE_GROUP else "세"


# 공식 연령 구간 (docs/02 §2.1).
#
# **유아기가 없는 것은 빠뜨린 것이 아니다.** 계약이 "유아기는 개월 수"로 받기로
# 했고 (docs/03 §3.1), 기준표도 개월 구간으로 나뉜다 (docs/02 §2.4). 만 5세는
# 60~71개월이라 두 구간에 걸쳐 세→개월 환산이 유일하지 않다. 그래서 `세` 로 온
# 유아기 나이는 되돌리지 않고 `None` 을 내어 호출자가 400 을 받게 한다 —
# 조회되지 않을 것을 연령대만 채워 200 으로 내보내는 것보다 정직하다.
_YEAR_BANDS: tuple[tuple[int, int, AgeGroup], ...] = (
    (11, 12, "유소년"),
    (13, 18, "청소년"),
    (19, 64, "성인"),
    (65, 200, "어르신"),
)
_MONTH_BAND = (48, 83)


def resolve_age_group(age: int, age_unit: AgeUnit) -> AgeGroup | None:
    """계약이 응답에 실을 연령대 (docs/03 §3.2).

    점수를 낼 수 있는지와는 다른 물음이다 — 어르신과 만 7~10세는 연령대가 정해지되
    `factors` 가 빈 배열로 나간다 (docs/03 §2.4).

    **`세` 로 온 만 4~6세는 `None` 이다** — 유아기는 개월로 받는다 (docs/03 §3.1).

    **만 7~10세는 공식 구간에 없다.** 측정이 0건이라 그렇다 (docs/02 §2.2). 가장
    가까운 `유소년` 으로 둔다 — 확정 전 기본값이다 (docs/02 §6 ④).
    """
    if age_unit == "개월":
        return "유아기" if _MONTH_BAND[0] <= age <= _MONTH_BAND[1] else None
    for lo, hi, group in _YEAR_BANDS:
        if lo <= age <= hi:
            return group
    return "유소년" if 7 <= age <= 10 else None
