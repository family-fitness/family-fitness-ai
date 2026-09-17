"""미션 문구를 LLM 이 쓴다 — 켜면.

기본은 꺼짐이다(.env 의 COACH_LLM).
어느 순서로 할지는 처방 자료와 클립 표가 정한다.

지어내는 것은 두 겹으로 막는다. 프롬프트로 시키고, 돌아온 문장을 다시 잰다
(길이·금지 어휘·개수). 어긋나면 통째로 버린다. 반쯤 고쳐 쓰지 않는다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from family_fitness_ai.common import copy as words
from family_fitness_ai.common import llm
from family_fitness_ai.common.settings import settings

if TYPE_CHECKING:  # pragma: no cover
    from family_fitness_ai.coach.compose import Constraints

log = logging.getLogger(__name__)

SYSTEM = """너는 가족 운동 앱의 문구를 쓴다. 편성은 이미 끝났다 — 너는 정해진
운동에 붙일 말만 쓴다.

- 한국어. 아이 문장은 초등학생이 읽는 말로, 부모 문장은 담백하게.
- 주어진 운동명·체력요인 말고 다른 운동이나 효과를 지어내지 않는다.
- 숫자를 새로 만들지 않는다. 주어진 횟수·분만 쓴다.
- 의학적 주장을 하지 않는다. 진단·치료·통증·체중을 말하지 않는다.
- 이 말들을 쓰지 않는다: 부족, 미달, 하위, 비만, 저체중, 열등.
- title 은 16자 이내, child 는 45자 이내, parent 는 70자 이내.
- 물음표로 끝나는 권유는 아이 문장에만 쓴다."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "missions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "child": {"type": "string"},
                    "parent": {"type": "string"},
                },
                "required": ["title", "child", "parent"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["missions"],
    "additionalProperties": False,
}

LIMITS = {"title": 16, "child": 45, "parent": 70}


@dataclass
class Written:
    #: LLM 을 부르려 했나. 꺼져 있으면 False.
    used: bool
    #: ok · failed · off
    status: str


def enabled() -> bool:
    return settings().coach_llm and llm.available()


def _acceptable(row: dict[str, Any]) -> bool:
    for key, limit in LIMITS.items():
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            return False
        if words.banned_words_in(value):
            return False
    return True


def write_copy(
    missions: list[dict[str, Any]],
    factor: str,
    band: str,
    age_group: str,
    constraints: Constraints,
) -> Written:
    """미션의 title·copy 를 LLM 문장으로 바꾼다. 실패하면 규칙 문구가 남는다."""
    if not enabled() or not missions:
        return Written(used=False, status="off")

    brief = [
        {
            "index": index,
            "기간": mission["period"]["start_date"],
            "운동": [
                f"[{session['phase']}] {session['exercise_name']}"
                for session in mission["sessions"]
            ],
            "분": mission["duration_min"],
            "역할": [p["role"] for p in mission["participants"]],
        }
        for index, mission in enumerate(missions)
    ]
    prompt = json.dumps(
        {
            "연령대": age_group,
            "대상_체력요인": factor or None,
            "요인_상태": words.BAND_COPY.get(band, "확인 전"),
            "주당_횟수": constraints.days_per_week,
            "미션": brief,
            "요청": "미션마다 title·child·parent 를 미션 순서대로 하나씩 쓴다",
        },
        ensure_ascii=False,
    )

    payload = llm.ask_json_or_none(prompt, SCHEMA, SYSTEM, max_tokens=4000)
    if payload is None:
        return Written(used=True, status="failed")

    rows = payload.get("missions", [])
    if len(rows) != len(missions) or not all(_acceptable(row) for row in rows):
        log.warning("문구가 규칙에 어긋난다 — 규칙 문구를 쓴다")
        return Written(used=True, status="failed")

    for mission, row in zip(missions, rows, strict=True):
        mission["title"] = row["title"].strip()
        mission["copy"] = {
            "child": row["child"].strip(),
            "parent": row["parent"].strip(),
        }
    return Written(used=True, status="ok")


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "grounded": {"type": "boolean"},
    },
    "required": ["answer", "grounded"],
    "additionalProperties": False,
}

ANSWER_SYSTEM = """너는 가족 운동 앱의 질문에 답한다. 주어진 자료만 가지고 답한다.

- 두세 문장. 문장 끝마다 근거 번호를 [1] 처럼 붙인다.
- 자료에 없는 말은 하지 않는다. 자료로 답할 수 없으면 grounded 를 false 로 둔다.
- 진단·치료·약·체중을 말하지 않는다.
- 이 말들을 쓰지 않는다: 부족, 미달, 하위, 비만, 저체중, 열등."""


def write_answer(question: str, passages: list[tuple[int, str]]) -> str | None:
    """근거 문단으로 답을 쓴다. 쓸 수 없으면 None — 부르는 쪽이 거부한다."""
    if not enabled() or not passages:
        return None

    prompt = json.dumps(
        {
            "질문": question,
            "자료": [{"번호": index, "내용": text} for index, text in passages],
        },
        ensure_ascii=False,
    )
    payload = llm.ask_json_or_none(prompt, ANSWER_SCHEMA, ANSWER_SYSTEM, max_tokens=1500)
    if not payload:
        return None

    answer = str(payload.get("answer", "")).strip()
    if not payload.get("grounded") or not answer or words.banned_words_in(answer):
        return None
    return answer


# ── 한 주 편성 ────────────────────────────────────────────────────────────

PLAN_SYSTEM = """너는 가족 운동 코치다. 한 가족의 한 주 운동을 짠다.

고르는 자리
- **주어진 클립 목록 안에서만** 고른다. 목록에 없는 id 를 쓰면 그 날은 버려진다.
- 날마다 준비운동 → 본운동 → 정리운동 순서로 쌓는다. 준비와 정리는 한두 개,
  본운동이 가운데를 채운다. 한 회 길이가 요청한 분에 가깝게 맞춘다.
- 같은 동작을 한 주에 두 번 이상 넣지 않는다. 날마다 달라야 아이가 지루해하지
  않는다.
- 대상 체력요인에 맞는 동작과, 근거에 실제로 나온 운동 이름을 앞세운다.
- 연령대가 다른 클립도 고를 수 있다. 다만 그 아이가 하기에 무리인 동작은 넣지
  않는다.

쓰는 말
- 한국어. child 는 초등학생이 읽는 말로, parent 는 담백하게.
- reason 에는 왜 이렇게 짰는지 한두 문장을 쓰고, 근거 번호를 [1] 처럼 붙인다.
  근거에 없는 효과를 지어내지 않는다.
- 진단·치료·통증·체중을 말하지 않는다.
- 이 말들을 쓰지 않는다: 부족, 미달, 하위, 비만, 저체중, 열등.
- title 은 16자, child 는 45자, parent 는 70자 안이다."""

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day_offset": {"type": "integer"},
                    "title": {"type": "string"},
                    "child": {"type": "string"},
                    "parent": {"type": "string"},
                    "reason": {"type": "string"},
                    "clips": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "phase": {
                                    "type": "string",
                                    "enum": ["준비운동", "본운동", "정리운동"],
                                },
                            },
                            "required": ["id", "phase"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["day_offset", "title", "child", "parent", "reason", "clips"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["days"],
    "additionalProperties": False,
}


def plan_week(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """한 주 편성을 LLM 에게 맡긴다. 못 받으면 None — 부르는 쪽이 규칙으로 짠다."""
    if not enabled():
        return None
    answer = llm.ask_json_or_none(
        json.dumps(payload, ensure_ascii=False),
        PLAN_SCHEMA,
        PLAN_SYSTEM,
        max_tokens=8000,
        timeout=45.0,
    )
    if not answer:
        return None
    days = answer.get("days")
    return days if isinstance(days, list) and days else None
