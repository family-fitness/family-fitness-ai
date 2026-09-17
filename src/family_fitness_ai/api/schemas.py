"""요청 모양. 응답은 계약 문서의 payload 를 그대로 낸다 — 봉투를 씌우지 않는다."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from family_fitness_ai.common.errors import item_not_allowed
from family_fitness_ai.common.items import BLOOD_PRESSURE

Sex = Literal["M", "F"]
AgeUnit = Literal["세", "개월"]
AgeGroup = Literal["유아기", "유소년", "청소년", "성인", "어르신"]
Role = Literal["주행자", "동반자", "응원"]

Measurements = dict[str, float]


def _no_blood_pressure(values: Measurements | None) -> Measurements | None:
    if not values:
        return values
    found = [code for code in BLOOD_PRESSURE if code in values]
    if found:
        raise item_not_allowed(found)
    return values


class ProfileIn(BaseModel):
    profile_ref: str
    age: Annotated[int, Field(ge=0, le=1200)]
    age_unit: AgeUnit = "세"
    sex: Sex
    height_cm: float | None = None
    weight_kg: float | None = None
    measurements: Measurements | None = None

    @field_validator("measurements")
    @classmethod
    def _check(cls, value: Measurements | None) -> Measurements | None:
        return _no_blood_pressure(value)


class TrajectoryIn(ProfileIn):
    item_code: str = "028"
    horizon_years: Annotated[int, Field(ge=1, le=10)] = 10


class VideoSearchIn(BaseModel):
    age_group: AgeGroup
    fitness_factors: list[str] = Field(default_factory=list)
    exercise_names: list[str] = Field(default_factory=list)
    k: Annotated[int, Field(ge=1, le=20)] = 5


class RunProfileIn(BaseModel):
    ref: str
    role: Role
    age: Annotated[int, Field(ge=0, le=1200)]
    age_unit: AgeUnit = "세"
    sex: Sex
    input_level: Literal["L0", "L1", "L2"] = "L0"
    height_cm: float | None = None
    weight_kg: float | None = None
    measurements: Measurements | None = None

    @field_validator("measurements")
    @classmethod
    def _check(cls, value: Measurements | None) -> Measurements | None:
        return _no_blood_pressure(value)


class PeriodIn(BaseModel):
    start_date: date
    weeks: Annotated[int, Field(ge=1, le=4)] = 1


class ConstraintsIn(BaseModel):
    days_per_week: Annotated[int, Field(ge=1, le=7)] = 3
    minutes_per_session: Annotated[int, Field(ge=5, le=60)] = 15
    #: 아랫집이 신경 쓰이면 뛰는 동작을 뺀다.
    quiet: bool = False
    #: 거실만큼 좁은 곳에서 할 수 있는 것만.
    small_space: bool = False
    #: 도구 없이 몸으로만.
    no_props: bool = False


class RunIn(BaseModel):
    profile_refs: Annotated[list[RunProfileIn], Field(min_length=1, max_length=4)]
    period: PeriodIn
    constraints: ConstraintsIn = Field(default_factory=ConstraintsIn)


class MessageIn(BaseModel):
    profile_ref: str = ""
    age_group: AgeGroup | None = None
    question: Annotated[str, Field(min_length=1, max_length=500)]
