"""LLM 부르는 자리 한 곳.

백엔드는 claude 와 gemini 둘이다(.env 의 COACH_BACKEND). 어느 쪽이든 **JSON 스키마를
주고 JSON 을 받는다** — 자유 문장을 받아 우리가 파싱하지 않는다. 형식이 어긋날
길을 모델 쪽에서 막는 편이 우리 쪽에서 정규식으로 줍는 것보다 낫다.

여기는 부르기만 한다. 무엇을 시킬지와 돌아온 것을 믿을지는 부르는 쪽이 정한다.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from family_fitness_ai.common.errors import temporarily_unavailable
from family_fitness_ai.common.settings import settings

log = logging.getLogger(__name__)

CLAUDE_DEFAULT = "claude-opus-5"
GEMINI_DEFAULT = "gemini-flash-latest"


def available() -> bool:
    """부를 키가 있나. 켜고 끄는 것은 부르는 쪽이 본다."""
    config = settings()
    key = config.gemini_api_key if config.coach_backend == "gemini" else config.anthropic_api_key
    return bool(key)


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
    backend = backend or config.coach_backend
    model = model or config.coach_model

    def call(which: str) -> dict[str, Any]:
        if which == "gemini":
            return _gemini(prompt, schema, system, model or GEMINI_DEFAULT, max_tokens)
        return _claude(prompt, schema, system, model or CLAUDE_DEFAULT, max_tokens, timeout)

    # 고른 백엔드로 먼저, 그래도 안 되면 다른 백엔드로 한 번. 한쪽이 붐빈다고
    # 서비스가 멈추지는 않는다.
    other = "claude" if backend == "gemini" else "gemini"
    ordered = [backend, other] if _has_key(other) else [backend]
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
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        # 짧은 문구 몇 줄에 오래 생각할 일이 아니다. 한 주 편성이 60초 안에 끝나야 한다.
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": schema},
        },
    )
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


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
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_schema": _openapi(schema),
            "max_output_tokens": max_tokens,
        },
    )
    return json.loads(response.text or "{}")


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
    "ask_json",
    "ask_json_or_none",
    "available",
    "temporarily_unavailable",
]
