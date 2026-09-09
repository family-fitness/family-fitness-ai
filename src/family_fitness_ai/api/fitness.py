"""`POST /v1/fitness/assessment` (docs/dev/AI-4).

이미 있는 계산을 계약 모양으로 내보낸다. 새로 하는 일은 `child_scope`/`parent_scope`
분리와 문구 생성뿐이고, **LLM 호출이 없다** (docs/01 §3.1).
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter

from ..common import copy as C
from ..common import logging as reqlog
from ..common.errors import ApiError, ErrorCode
from ..common.settings import RELEASE_DIR
from ..common.types import AgeGroup, InputLevel, resolve_age_group
from ..stats import assess as A
from ..stats.items import item_label
from ..stats.score import MIN_SAMPLE
from .schemas import (
    AssessmentRequest,
    AssessmentResponse,
    ChildScope,
    FactorScore,
    FocusOne,
    ParentCopy,
    ParentScope,
    PeerGrade,
)

router = APIRouter(prefix="/v1/fitness", tags=["fitness"])


@lru_cache(maxsize=1)
def reference() -> A.Reference:
    """산출물은 기동 시 한 번 올린다. 요청마다 CSV를 읽지 않는다 (docs/dev/AI-4 §1)."""
    return A.Reference(RELEASE_DIR)


@router.post("/assessment", response_model=AssessmentResponse)
def assessment(request: AssessmentRequest) -> AssessmentResponse:
    age_group = resolve_age_group(request.age, request.age_unit)
    if age_group is None:
        raise ApiError(
            ErrorCode.BAD_REQUEST,
            f"연령대를 정할 수 없다: {request.age}{request.age_unit}",
        )

    result = A.assess(
        reference(),
        age=request.age,
        age_unit=request.age_unit,
        sex=request.sex,
        measurements=request.measurements,
    )
    factors = [_factor(age_group, f) for f in result.factors]
    scored = [f for f in factors if f.score is not None]
    level = _input_level(request, scored)

    reqlog.add_fields(
        profile_ref=request.profile_ref,
        input_level=level,
        age_group=age_group,
        factor_count=len(scored),
        graded=result.grade is not None,
    )
    return AssessmentResponse(
        input_level=level,
        age_group=age_group,
        child_scope=_child(age_group, scored),
        parent_scope=ParentScope(
            grade=result.grade if level == "L2" else None,
            peer_distribution=[
                PeerGrade(grade=g, ratio=r)
                for g, r in reference().peer_distribution(age_group, request.age, request.sex)
            ],
            factors=factors if level == "L2" else [],
            text=_parent_copy(scored),
        ),
        low_sample=result.low_sample,
    )


def _input_level(request: AssessmentRequest, scored: list[FactorScore]) -> InputLevel:
    """**`measurements` 가 왔다고 `L2` 가 아니다** (docs/dev/AI-4 §2).

    그 연령 구간의 기준항목이 아니면 점수가 안 나온다. 하나도 안 나오면 `L1` 이다.
    """
    if scored:
        return "L2"
    return "L1" if request.height_cm is not None and request.weight_kg is not None else "L0"


def _factor(age_group: AgeGroup, f: A.FactorScore) -> FactorScore:
    """`n < 30` 이면 `score`·`percentile` 을 `null` 로 내린다.

    **API 경계에서 내린다** (docs/dev/AI-4 §4). 산출물이나 `stats` 안에서 지우면
    진단 CLI도 함께 눈이 멀고, `docs/02` §6 ③ 이 반대로 확정되면 되돌릴 값이 없다.
    """
    low = f.n < MIN_SAMPLE
    return FactorScore(
        factor=f.factor,
        item_code=f.item_code,
        item_name=f.item_name,
        item_label=item_label(age_group, f.item_code),
        unit=f.unit,
        value=f.value,
        score=None if low else f.score,
        percentile=None if low else f.percentile,
        band=None if low else f.band,
        n=f.n,
    )


def _child(age_group: AgeGroup, scored: list[FactorScore]) -> ChildScope:
    """요인 **하나**뿐이다. 요인 간 비교·순위를 담지 않는다 (docs/03 §3.3)."""
    if scored:
        focus = min(scored, key=lambda f: f.score if f.score is not None else 101.0).factor
    else:
        # 점수가 없으므로 요인을 고르는 근거도 없다 — 연령대 고정 제안이다
        # (docs/dev/AI-4 §3.1 · §6 ②).
        focus = C.AGE_GROUP_FOCUS[age_group]
    return ChildScope(focus_one=FocusOne(factor=focus, text=C.child_focus(focus)))


def _parent_copy(scored: list[FactorScore]) -> ParentCopy:
    """`strength` 는 잘하고 있는 요인 중 최고, `focus` 는 최저 (docs/dev/AI-4 §3.3)."""
    if not scored:
        return ParentCopy(
            strength="측정값을 넣으면 요인별로 살펴볼 수 있습니다",
            focus="측정값을 넣으면 이번에 키우기 좋은 영역을 알려드립니다",
        )
    ranked = sorted(scored, key=lambda f: f.score or 0.0)
    lowest = ranked[0]
    strong = [f for f in ranked if f.band == "strength"]
    top = strong[-1] if strong else ranked[-1]
    return ParentCopy(
        strength=C.parent_line(top.factor, top.band or "steady"),
        focus=C.parent_line(lowest.factor, lowest.band or "steady"),
    )
