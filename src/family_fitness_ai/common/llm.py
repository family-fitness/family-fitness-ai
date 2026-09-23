"""LLM 부르는 자리 한 곳.

백엔드는 claude 와 gemini 둘이다. 어느 쪽이든 **JSON 스키마를 주고 JSON 을 받는다** —
형식이 어긋날 길을 모델 쪽에서 막는 편이 우리 쪽에서 줍는 것보다 낫다. 그래도
울타리(```)를 씌워 보내는 모델이 있어 마지막에 한 번 더 걷어 낸다.

**모델을 바꿔도 어긋나지 않는다.** `.env` 의 `COACH_MODEL` 만 고쳐도 백엔드가 따라
간다(이름이 `gemini…` 면 제미나이, `claude…` 면 클로드). 모델마다 받는 것이 달라서
`effort` 처럼 어떤 모델이 거절하는 설정은 한 번 거절당하면 그 모델에 다시 보내지
않는다 — 모델 이름을 표로 박아 두면 새 모델이 나올 때마다 표가 낡는다.

여기는 부르기만 한다. 무엇을 시킬지와 돌아온 것을 믿을지는 부르는 쪽이 정한다.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

from family_fitness_ai.common.errors import temporarily_unavailable
from family_fitness_ai.common.settings import settings

log = logging.getLogger(__name__)

CLAUDE_DEFAULT = "claude-opus-5"
GEMINI_DEFAULT = "gemini-flash-latest"


def available() -> bool:
    """부를 키가 하나라도 있나.

    고른 백엔드가 붐비면 다른 쪽으로 넘어가므로 둘 중 하나만 있어도 된다. 켜고
    끄는 것은 부르는 쪽이 본다.
    """
    config = settings()
    return bool(config.anthropic_api_key or config.gemini_api_key)


def backend_for(model: str, fallback: str) -> str:
    """모델 이름이 어느 백엔드의 것인지 말해 준다."""
    name = model.strip().lower()
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith(("claude", "anthropic.")):
        return "claude"
    return fallback


_FENCE = re.compile(r"\A```[a-zA-Z]*\n|\n```\Z")


def loads(text: str) -> dict[str, Any]:
    """돌아온 글에서 JSON 하나. 울타리를 씌우거나 앞뒤에 말을 붙이는 모델이 있다."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()
    try:
        return dict(json.loads(text))
    except json.JSONDecodeError:
        first, last = text.find("{"), text.rfind("}")
        if first == -1 or last <= first:
            raise
        return dict(json.loads(text[first : last + 1]))


#: 붐빌 때 503 이 온다. 한 번은 더 해 보되 오래 매달리지 않는다 — 편성 전체가
#: 60초 안에 끝나야 해서, 여기서 다 쓰면 규칙 편성으로 내려갈 짬도 없다.
RETRIES = 2
#: 방금 넘어진 백엔드는 이만큼 쉬게 둔다. 한 번 붐비면 그 다음 부름도 거의
#: 붐비는데, 부를 때마다 같은 곳에서 두 번씩 기다리면 편성이 제 시간에 못 끝난다.
COOLDOWN_SEC = 120

_resting: dict[str, float] = {}


def _awake(backend: str) -> bool:
    return _resting.get(backend, 0) < time.time()


def ask_json(
    prompt: str,
    schema: dict[str, Any],
    system: str,
    *,
    max_tokens: int = 4000,
    timeout: float = 30.0,
    backend: str = "",
    model: str = "",
) -> dict[str, Any]:
    """스키마에 맞는 JSON 하나. 끝까지 실패하면 예외를 그대로 올린다."""
    config = settings()
    model = model or config.coach_model
    backend = backend or backend_for(model, config.coach_backend)
    # .env 에 못 보던 이름이 적혀 있어도 멈추지 않는다 — 제미나이가 아니면 클로드다.
    backend = "gemini" if backend.strip().lower() == "gemini" else "claude"

    def model_for(which: str) -> str:
        """그 백엔드에 보낼 모델. 다른 백엔드의 모델 이름은 쓰지 않는다."""
        if model and backend_for(model, which) == which:
            return model
        return GEMINI_DEFAULT if which == "gemini" else CLAUDE_DEFAULT

    def call(which: str) -> dict[str, Any]:
        if which == "gemini":
            return _gemini(prompt, schema, system, model_for(which), max_tokens)
        return _claude(prompt, schema, system, model_for(which), max_tokens, timeout)

    # 고른 백엔드로 먼저, 그래도 안 되면 다른 백엔드로 한 번. 한쪽이 붐빈다고
    # 서비스가 멈추지는 않는다.
    other = "claude" if backend == "gemini" else "gemini"
    # 키 없는 쪽은 아예 빼 둔다. 고른 쪽에 키가 없는데 다른 쪽에 있으면 그리로 간다.
    ordered = [which for which in (backend, other) if _has_key(which)] or [backend]
    # 방금 넘어진 쪽은 뒤로 미룬다. 둘 다 쉬는 중이면 원래 차례대로 간다.
    awake = [which for which in ordered if _awake(which)]
    if awake:
        ordered = awake + [which for which in ordered if which not in awake]
    plan = [ordered[0]] * RETRIES + ordered[1:]

    last: Exception | None = None
    for attempt, which in enumerate(plan):
        try:
            answer = call(which)
            _resting.pop(which, None)
            return answer
        except Exception as error:  # noqa: BLE001 — 무엇이 넘어졌든 다음 차례로 간다
            last = error
            _resting[which] = time.time() + COOLDOWN_SEC
            log.warning("LLM 호출 실패(%s) — %s", which, type(error).__name__)
            if attempt < len(plan) - 1:
                time.sleep(min(2**attempt, 4) * 0.5 + random.random() * 0.5)
    raise last if last else RuntimeError("LLM 호출 실패")


def _has_key(backend: str) -> bool:
    config = settings()
    return bool(config.gemini_api_key if backend == "gemini" else config.anthropic_api_key)


#: 「오래 생각하지 마라」를 받지 않는 모델이 있다 — 클로드는 하이쿠와 한 세대 전
#: 소네트가, 제미나이는 생각을 끌 수 없는 모델이 그렇다. 400 을 한 번 받으면 여기
#: 적어 두고 그 모델에는 다시 보내지 않는다.
_NO_EFFORT: set[str] = set()


def _claude(
    prompt: str,
    schema: dict[str, Any],
    system: str,
    model: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings().anthropic_api_key, timeout=timeout)

    def call(with_effort: bool) -> Any:
        # Any 로 둔다 — SDK 의 TypedDict 는 effort 를 빼고 조립하는 길을 받지 않는다.
        output_config: Any = {"format": {"type": "json_schema", "schema": schema}}
        if with_effort:
            # 짧은 문구 몇 줄에 오래 생각할 일이 아니다. 한 주 편성이 60초 안에 끝나야 한다.
            output_config["effort"] = "low"
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config=output_config,
        )

    try:
        response = call(model not in _NO_EFFORT)
    except anthropic.BadRequestError as error:
        if model in _NO_EFFORT or "effort" not in str(error):
            raise
        log.info("%s 는 effort 를 받지 않는다 — 빼고 다시 부른다", model)
        _NO_EFFORT.add(model)
        response = call(False)

    text = next((block.text for block in response.content if block.type == "text"), "")
    return loads(text)


def _openapi(schema: Any) -> Any:
    """제미나이는 OpenAPI 부분집합만 받는다 — additionalProperties 를 모른다."""
    if isinstance(schema, dict):
        return {
            key: _openapi(value) for key, value in schema.items() if key != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_openapi(item) for item in schema]
    return schema


def _gemini(
    prompt: str, schema: dict[str, Any], system: str, model: str, max_tokens: int
) -> dict[str, Any]:
    from google import genai

    # 제미나이 SDK 는 스스로도 여러 번 다시 부른다. 그 재시도까지 겹치면 한 번
    # 부르는 데 1분이 넘어가므로, 짧게 끊고 우리 쪽에서 다음 차례로 넘긴다.
    client = genai.Client(
        api_key=settings().gemini_api_key,
        http_options={"timeout": 20_000, "retry_options": {"attempts": 1}},
    )

    def call(quick: bool) -> Any:
        # Any 로 둔다 — SDK 의 TypedDict 는 칸을 빼고 조립하는 길을 받지 않는다.
        config: Any = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_schema": _openapi(schema),
            "max_output_tokens": max_tokens,
        }
        if quick:
            # 생각도 max_output_tokens 에서 깎인다. 켜 두면 짧은 문구를 물었을 때
            # 생각만 하다 답을 못 내고 빈 글이 돌아온다 — 클로드의 effort:low 자리다.
            config["thinking_config"] = {"thinking_budget": 0}
        return client.models.generate_content(model=model, contents=prompt, config=config)

    try:
        response = call(model not in _NO_EFFORT)
    except Exception as error:
        if model in _NO_EFFORT or "thinking" not in str(error).lower():
            raise
        log.info("%s 는 생각을 끌 수 없다 — 그대로 부른다", model)
        _NO_EFFORT.add(model)
        response = call(False)

    return loads(response.text or "")


def ask_json_or_none(
    prompt: str, schema: dict[str, Any], system: str, **kwargs: Any
) -> dict[str, Any] | None:
    """부르다 넘어져도 서비스는 계속 간다 — 부르는 쪽이 규칙으로 내려갈 수 있게."""
    try:
        return ask_json(prompt, schema, system, **kwargs)
    except Exception:  # noqa: BLE001 — 어떤 실패든 규칙으로 내려간다
        log.warning("LLM 호출 실패", exc_info=True)
        return None


__all__ = [
    "CLAUDE_DEFAULT",
    "GEMINI_DEFAULT",
    "backend_for",
    "loads",
    "ask_json",
    "ask_json_or_none",
    "available",
    "temporarily_unavailable",
]
