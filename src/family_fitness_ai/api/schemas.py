"""`POST /v1/fitness/assessment` 의 요청·응답 (docs/03 §3).

**검증은 요청 모델에 있다.** 라우터 안에서 검사하면 엔드포인트마다 빠뜨린다
(docs/dev/AI-3 §2.3).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..common.errors import ApiError, ErrorCode
from ..common.types import (
    BLOCKED_ITEM_CODES,
    AgeGroup,
    AgeUnit,
    Band,
    FitnessFactor,
    InputLevel,
    Sex,
)

# docs/03 §3.2 — 상시 노출 문구. 그대로 표시한다.
DISCLAIMER = (
    "국민체력100 측정 데이터를 바탕으로 한 참고 정보입니다. "
    "질병의 진단·치료를 위한 것이 아니며, 건강에 관한 판단은 전문가와 상담하세요."
)


class AssessmentRequest(BaseModel):
    profile_ref: str
    age: int
    age_unit: AgeUnit
    sex: Sex
    height_cm: Annotated[float, Field(ge=30.0, le=230.0)] | None = None
    weight_kg: Annotated[float, Field(ge=5.0, le=250.0)] | None = None
    measurements: dict[str, float] = Field(default_factory=dict)

    @field_validator("measurements")
    @classmethod
    def _no_blood_pressure(cls, value: dict[str, float]) -> dict[str, float]:
        """혈압은 값이 있어도 받지 않는다 (docs/03 §9). 혈압 해석은 질병 영역이다."""
        blocked = sorted(BLOCKED_ITEM_CODES & value.keys())
        if blocked:
            raise ApiError(
                ErrorCode.ITEM_NOT_ALLOWED,
                f"AI는 이 항목을 다루지 않는다: {', '.join(blocked)}",
            )
        return value


class FocusOne(BaseModel):
    """`copy` 는 계약의 키다 (docs/03 §3.3). `BaseModel.copy` 와 겹쳐 별칭으로 둔다."""

    model_config = ConfigDict(populate_by_name=True)

    factor: FitnessFactor
    text: str = Field(serialization_alias="copy", validation_alias="copy")


class ChildScope(BaseModel):
    """**점수·백분위·체중이 없다** (docs/03 §2.6).

    아이 화면이 실수로 노출할 값이 애초에 페이로드에 없게 한다. 필드를 더할 때
    이 문장을 먼저 읽는다.
    """

    focus_one: FocusOne | None


class PeerGrade(BaseModel):
    grade: str
    ratio: float


class FactorScore(BaseModel):
    factor: FitnessFactor
    item_code: str
    item_name: str
    item_label: str
    unit: str
    value: float | None
    score: float | None
    percentile: int | None
    band: Band | None
    n: int


class ParentCopy(BaseModel):
    strength: str
    focus: str


class ParentScope(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    grade: str | None
    peer_distribution: list[PeerGrade]
    factors: list[FactorScore]
    text: ParentCopy = Field(serialization_alias="copy", validation_alias="copy")


class AssessmentResponse(BaseModel):
    input_level: InputLevel
    age_group: AgeGroup
    child_scope: ChildScope
    parent_scope: ParentScope
    low_sample: bool
    disclaimer: str = DISCLAIMER
