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
