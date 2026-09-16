"""`/v1/coach/runs` 와 `/v1/coach/messages` 의 요청·응답 모양 (docs/03 §5·§6).

**여기는 모양만 있다. 라우터는 다음 단계다.**

필드 이름의 정본은 백엔드 와이어다 —
`backend/.../shared/ai/AiWire.kt` 의 `@JsonProperty("...")`. 그래서 모든 필드에
`alias` 를 붙이고 `scripts/wire_check.py` 가 그 이름 집합을 기계로 맞춘다.
응답을 낼 때는 `model_dump(by_alias=True, mode="json")` 을 쓴다.

**와이어에 없는 이름을 내보내지 않는다.** `end_sec`(영상 끝) · `week` · `cadence`
(주기) 자리가 백엔드에 없다. 일일/주간은 `period` 의 길이로 가른다 — 하루면 일일,
이레면 주간이고 계약은 바뀌지 않는다.

**요청은 계약보다 넓게 받는다.** docs/03 §5.1 은 `profile_refs` 를 1~4명이라
적었지만 백엔드는 가족 전원을 보내고, `measurements` 는 빈 객체가 아니라 `null` 을
보낸다. 계약 표를 그대로 검사하면 실제 요청이 400 이 된다. 호출자가 고칠 수 없는
400 을 만들지 않는다.

**응답은 우리가 내는 값이라 닫는다.** `status` 와 단계 이름은 Literal 이다. 오타를
mypy 가 막는다.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

# ── 백엔드가 자르는 자리 ──────────────────────────────────────────────
#
# 전부 /Users/yubeom-ig/temp/family-fitness-be 에서 직접 확인한 값이다.
#   coach_runs.ai_run_id            varchar(40)   V1__init.sql:216
#   coach_runs.ai_refusal_reason    varchar(60)   V130__coaching_refusal_and_coach_role.sql:4
#                                                 (CoachRun.MAX_REFUSAL_REASON = 60)
#   coach_messages.refusal_reason   varchar(40)   V1__init.sql:315
#   missions.title                  varchar(120)  ProposalConverter.MAX_TITLE = 120
#   missions.copy_child/copy_parent varchar(400)  ProposalConverter.MAX_COPY = 400
MAX_RUN_ID = 40
MAX_TITLE = 120
MAX_COPY = 400
MAX_RUN_REFUSAL_REASON = 60
MAX_MESSAGE_REFUSAL_REASON = 40


def _cut(limit: int) -> Callable[[str], str]:
    """상한을 넘는 문구는 **우리가 미리 자른다.**

    자르지 않으면 백엔드가 `take()` 로 자르거나 컬럼이 넘쳐 저장이 실패한다.
    문구(title·copy·거부 사유)는 잘려도 뜻이 남으므로 자른다.

    **`run_id` 는 자르지 않고 거부한다** (`max_length`). 잘린 id 로는 폴링이
    되지 않아 조용히 잘못된 값을 넘기는 쪽이 더 나쁘다. 우리가 내는 id 는
    `cr_` + 16자 = 19자이므로 40자를 넘는 것은 우리 버그다 — 시끄럽게 터뜨린다.
    """

    def cut(value: str) -> str:
        return value[:limit]

    return cut


Title = Annotated[str, AfterValidator(_cut(MAX_TITLE))]
Copy = Annotated[str, AfterValidator(_cut(MAX_COPY))]
RunRefusalReason = Annotated[str, AfterValidator(_cut(MAX_RUN_REFUSAL_REASON))]

# docs/03 §5.3 — 우리가 내는 값이라 닫는다.
RunStatus = Literal["running", "succeeded", "failed", "refused"]
# docs/03 §5.4 — 4단계와 그 결과.
StepName = Literal["assess", "retrieve", "compose", "verify"]
StepStatus = Literal["ok", "partial", "failed"]
# docs/03 §6.2 — 메시지 거부 사유. 네 값 모두 40자 안이다.
MessageRefusalReason = Literal[
    "no_relevant_source", "age_filter_empty", "medical_query", "no_citation_generated"
]


class _Wire(BaseModel):
    """와이어 이름과 파이썬 이름을 둘 다 받는다. 낼 때는 `by_alias=True` 다."""

    model_config = ConfigDict(populate_by_name=True)


# ── 요청 ──────────────────────────────────────────────────────────────


class CoachProfileRef(_Wire):
    """`AiWire.CoachRunRequestBody.ProfileRefBody`.

    **`ref` 를 바이트 그대로 보관한다.** strip·소문자화·정규화를 하지 않는다 —
    백엔드 `ProposalConverter.refIndex` 는 우리가 돌려준 문자열로 프로필을 찾고,
    찾지 못한 참여자는 **조용히 버려진다**. 참여자가 0명이 된 항목은 미션이 아예
    생기지 않으므로, 우리가 공백 하나를 고치면 미션이 사라진다.

    `role`·`input_level`·`age_unit`·`sex` 는 **Literal 로 닫지 않는다.** 아는 값은
    아래에 적지만, 모르는 값이 와도 400 을 내지 않는다.
      role        주행자 · 동반자 · 응원      (CoachRun.kt `CoachRoles`)
      input_level L0 · L1 · L2                (AiGateway.kt:67-69)
      sex         M · F                       (shared/domain/Sex.kt)
      age_unit    세 · 개월                   (shared/domain/AgeGroup.kt:23)
    """

    ref: str = Field(alias="ref")
    role: str = Field(alias="role")
    age: int = Field(alias="age")
    age_unit: str = Field(alias="age_unit")
    sex: str = Field(alias="sex")
    input_level: str = Field(alias="input_level")
    height_cm: float | None = Field(default=None, alias="height_cm")
    weight_kg: float | None = Field(default=None, alias="weight_kg")
    # 백엔드는 측정값이 없으면 빈 객체가 아니라 null 을 보낸다
    # (AiWire.kt `measurements = a.measurements.takeIf { it.isNotEmpty() }`).
    measurements: dict[str, float] | None = Field(default=None, alias="measurements")


class CoachRunPeriod(_Wire):
    """`AiWire.CoachRunRequestBody.Period`. 백엔드는 `weeks = 1` 만 보낸다."""

    start_date: dt.date = Field(alias="start_date")
    weeks: Annotated[int, Field(ge=1, le=4)] = Field(default=1, alias="weeks")


class CoachConstraints(_Wire):
    """`AiWire.CoachRunRequestBody.Constraints`.

    상·하한은 백엔드가 이미 같은 값으로 검사한다
    (`StartCoachRunRequest` 의 `@Min(1) @Max(7)` · `@Min(5) @Max(60)`).
    """

    days_per_week: Annotated[int, Field(ge=1, le=7)] = Field(alias="days_per_week")
    minutes_per_session: Annotated[int, Field(ge=5, le=60)] = Field(alias="minutes_per_session")


class CoachRunRequest(_Wire):
    """`POST /v1/coach/runs` 의 본문 (docs/03 §5.1).

    **인원 상한을 검사하지 않는다.** 계약 표는 1~4명이라 적었지만 백엔드
    `CoachRunExecutor` 는 가족 구성원 전원을 보낸다. 5명 가족이 400 을 받으면
    호출자가 고칠 방법이 없다.
    """

    profile_refs: Annotated[list[CoachProfileRef], Field(min_length=1)] = Field(
        alias="profile_refs"
    )
    period: CoachRunPeriod = Field(alias="period")
    constraints: CoachConstraints = Field(alias="constraints")


class CoachMessageRequest(_Wire):
    """`POST /v1/coach/messages` 의 본문 (docs/03 §6.1).

    `age_group` 도 닫지 않는다 — 백엔드 `AgeGroup` 의 `@JsonValue` 는
    유아기·유소년·청소년·성인·어르신 이지만, 우리가 모르는 값에 400 을 내는 대신
    연령 필터에서 0건으로 거부하는 쪽이 낫다.
    """

    profile_ref: str = Field(alias="profile_ref")
    age_group: str = Field(alias="age_group")
    question: Annotated[str, Field(min_length=1, max_length=500)] = Field(alias="question")


# ── 응답 ──────────────────────────────────────────────────────────────


class Citation(_Wire):
    """`AiWire.CitationBody` (docs/03 §6.3).

    **`label` 은 필수이고 빈 문자열을 허용하지 않는다.** 백엔드
    `CitationBody.label: String` 에는 기본값이 없다 — 키가 빠지면 잭슨
    역직렬화가 터지고, 그 예외는 `HttpAiGateway` 가 잡는
    `RestClientResponseException`·`ResourceAccessException` 이 아니라서 503 으로
    번역되지 않는다. 코치 실행이 그냥 FAILED 가 된다. **이 파일에서 가장 중요한
    한 줄이다.**
    """

    index: Annotated[int, Field(ge=1)] = Field(alias="index")
    label: Annotated[str, Field(min_length=1)] = Field(alias="label")
    chunk_id: Annotated[str, Field(min_length=1)] = Field(alias="chunk_id")
    url: str | None = Field(default=None, alias="url")


class CoachRunAccepted(_Wire):
    """`POST /v1/coach/runs` 의 202 (docs/03 §5.2)."""

    run_id: Annotated[str, Field(min_length=1, max_length=MAX_RUN_ID)] = Field(alias="run_id")
    status: RunStatus = Field(default="running", alias="status")
    poll_after_ms: Annotated[int, Field(ge=0)] = Field(default=1500, alias="poll_after_ms")


class Step(_Wire):
    """`AiWire.CoachRunResultBody.StepBody` (docs/03 §5.4). 감사 기록이다."""

    seq: Annotated[int, Field(ge=1)] = Field(alias="seq")
    name: StepName = Field(alias="name")
    status: StepStatus = Field(alias="status")
    summary: str = Field(default="", alias="summary")


class Video(_Wire):
    """`AiWire.CoachRunResultBody.VideoBody`.

    **`end_sec` 자리가 백엔드에 없다.** 영상 구간의 끝은 보내지 않는다. 필드를
    더하고 싶으면 먼저 백엔드에 자리가 생겨야 한다 (AI-13 안건).
    """

    video_id: str = Field(alias="video_id")
    start_sec: int | None = Field(default=None, alias="start_sec")


class Session(_Wire):
    """`AiWire.CoachRunResultBody.SessionBody` (docs/03 §5.7).

    `video` 가 `null` 인 것은 결함이 아니다. `evidence` 는 최소 1개다 — 비면
    백엔드 `ProposalConverter` 가 "인용 전체"로 읽어 근거가 뭉개진다.
    `fitness_factor` 는 빈 문자열일 수 있다 (백엔드 기본값이 `""` 다).
    """

    day_offset: Annotated[int, Field(ge=0)] = Field(alias="day_offset")
    exercise_name: str = Field(alias="exercise_name")
    fitness_factor: str = Field(default="", alias="fitness_factor")
    duration_min: Annotated[int, Field(ge=0)] = Field(alias="duration_min")
    video: Video | None = Field(default=None, alias="video")
    evidence: Annotated[list[int], Field(min_length=1)] = Field(alias="evidence")


class Participant(_Wire):
    """`AiWire.CoachRunResultBody.ParticipantBody`. `ref` 는 요청 그대로다."""

    ref: str = Field(alias="ref")
    role: str = Field(default="", alias="role")


class MissionPeriod(_Wire):
    """`AiWire.CoachRunResultBody.PeriodBody`.

    **일일/주간을 가르는 유일한 자리다.** `start_date == end_date` 면 일일이고,
    엿새 차(7일)면 주간이다. 백엔드 `MissionService.MissionView` 에
    `startDate`·`endDate` 가 있어 프론트가 날짜로 가를 수 있다.
    """

    start_date: dt.date = Field(alias="start_date")
    end_date: dt.date = Field(alias="end_date")


class MissionCopy(_Wire):
    """`AiWire.CoachRunResultBody.CopyBody`.

    와이어의 키는 `copy` 다. `BaseModel.copy` 와 겹쳐 파이썬 이름은 `text` 로
    두고 별칭으로 맞춘다 (`api/schemas.py` 의 `FocusOne` 과 같은 방식).
    """

    child: Copy = Field(default="", alias="child")
    parent: Copy = Field(default="", alias="parent")


class Mission(_Wire):
    """`AiWire.CoachRunResultBody.MissionBody` (docs/03 §5.6).

    **주기(week·cadence) 자리가 없다.** 시간 축은 `period` 한 쌍뿐이다.
    """

    title: Title = Field(alias="title")
    period: MissionPeriod = Field(alias="period")
    participants: Annotated[list[Participant], Field(min_length=1)] = Field(alias="participants")
    sessions: Annotated[list[Session], Field(min_length=1)] = Field(alias="sessions")
    text: MissionCopy = Field(alias="copy")


class Proposal(_Wire):
    """`AiWire.CoachRunResultBody.ProposalBody` (docs/03 §5.5). 인용 최소 1개."""

    missions: Annotated[list[Mission], Field(min_length=1)] = Field(alias="missions")
    citations: Annotated[list[Citation], Field(min_length=1)] = Field(alias="citations")


class CoachRunResult(_Wire):
    """`GET /v1/coach/runs/{run_id}` 의 200 (docs/03 §5.3)."""

    run_id: Annotated[str, Field(min_length=1, max_length=MAX_RUN_ID)] = Field(alias="run_id")
    status: RunStatus = Field(alias="status")
    steps: list[Step] = Field(default_factory=list, alias="steps")
    proposal: Proposal | None = Field(default=None, alias="proposal")
    refused: bool = Field(default=False, alias="refused")
    refusal_reason: RunRefusalReason | None = Field(default=None, alias="refusal_reason")


class CoachMessageResponse(_Wire):
    """`POST /v1/coach/messages` 의 200 (docs/03 §6.2).

    `refusal_reason` 은 네 값으로 닫는다 — 전부 40자 안이라 백엔드
    `coach_messages.refusal_reason varchar(40)` 에 잘리지 않고 들어간다.
    """

    answer: str = Field(default="", alias="answer")
    citations: list[Citation] = Field(default_factory=list, alias="citations")
    refused: bool = Field(default=False, alias="refused")
    refusal_reason: MessageRefusalReason | None = Field(default=None, alias="refusal_reason")


# `scripts/wire_check.py` 가 쓴다. 백엔드 클래스 이름 → 우리 모델.
WIRE_CLASSES: dict[str, type[BaseModel]] = {
    "CoachRunRequestBody": CoachRunRequest,
    "ProfileRefBody": CoachProfileRef,
    "Period": CoachRunPeriod,
    "Constraints": CoachConstraints,
    "CoachRunAcceptedBody": CoachRunAccepted,
    "CoachRunResultBody": CoachRunResult,
    "StepBody": Step,
    "ProposalBody": Proposal,
    "MissionBody": Mission,
    "PeriodBody": MissionPeriod,
    "ParticipantBody": Participant,
    "SessionBody": Session,
    "VideoBody": Video,
    "CopyBody": MissionCopy,
    "CitationBody": Citation,
    "CoachMessageRequestBody": CoachMessageRequest,
    "CoachMessageBody": CoachMessageResponse,
}


def wire_names(model: type[BaseModel]) -> set[str]:
    """그 모델이 와이어에 내보내는 이름들."""
    return {field.alias or name for name, field in model.model_fields.items()}
