"""요청 모양. 응답은 계약 문서의 payload 를 그대로 낸다 — 봉투를 씌우지 않는다."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from family_fitness_ai.common.errors import item_not_allowed
from family_fitness_ai.common.items import BLOOD_PRESSURE, FACTORS

Sex = Literal["M", "F"]
AgeUnit = Literal["세", "개월"]
AgeGroup = Literal["유아기", "유소년", "청소년", "성인", "어르신"]
Role = Literal["주행자", "동반자", "응원"]

Measurements = dict[str, float]

#: recent_video_ids 를 몇 개까지 보나.
RECENT_LIMIT = 150


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
    #: 일간 — 주행자가 하루에 한 번씩.
    days_per_week: Annotated[int, Field(ge=1, le=7)] = 3
    minutes_per_session: Annotated[int, Field(ge=5, le=60)] = 15
    #: 주간 — 그 주 안에 한 번 길게, 온 가족이 함께. 비우면 만들지 않는다.
    weekly_minutes: Annotated[int, Field(ge=5, le=90)] | None = None
    #: 아랫집이 신경 쓰이면 뛰는 동작을 뺀다.
    quiet: bool = False
    #: 거실만큼 좁은 곳에서 할 수 있는 것만.
    small_space: bool = False
    #: 도구 없이 몸으로만.
    no_props: bool = False
    #: 보호자가 키워 주고 싶은 역량. 여덟 요인의 한글 이름(예: 유연성). 있으면 측정으로
    #: 고른 가장 낮은 요인보다 먼저 쓴다. 비우면 지금처럼 측정으로 고른다.
    focus_factor: str | None = None
    #: 보호자도 같이 한다. 참여자를 늘리지는 않는다 — LLM 코치에게 알려 문구에만 반영한다.
    with_companion: bool = False
    #: 그 사람이 최근 14일 동안 미션으로 받은 영상 id(유튜브 id 또는 공단 파일 이름),
    #: 최근 것부터. 후보를 고를 때 뒤로 미룬다. 모르는 id 는 고를 때 걸리지 않을 뿐이다.
    recent_video_ids: list[str] = Field(default_factory=list)

    @field_validator("recent_video_ids")
    @classmethod
    def _recent_first(cls, value: list[str]) -> list[str]:
        # 약속은 최근 것부터 150개까지다. 더 오면 거절하지 않고 앞의 150개만 본다 —
        # 편성이 통째로 떨어지는 것보다 오래된 몇 개를 덜 미루는 편이 낫다.
        return [video_id for video_id in value if video_id][:RECENT_LIMIT]

    @field_validator("focus_factor")
    @classmethod
    def _known_factor(cls, value: str | None) -> str | None:
        # 빈 글자는 안 고른 것으로 본다. 모르는 요인은 받지 않는다 — 조용히 버리면
        # 보호자가 고른 것이 편성에 안 들어갔는데도 들어간 줄 안다.
        if not value:
            return None
        if value not in FACTORS:
            raise ValueError(f"{', '.join(FACTORS)} 중 하나여야 합니다")
        return value


class RunIn(BaseModel):
    profile_refs: Annotated[list[RunProfileIn], Field(min_length=1, max_length=4)]
    period: PeriodIn
    constraints: ConstraintsIn = Field(default_factory=ConstraintsIn)


class MessageIn(BaseModel):
    profile_ref: str = ""
    age_group: AgeGroup | None = None
    question: Annotated[str, Field(min_length=1, max_length=500)]
