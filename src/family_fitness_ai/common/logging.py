"""한 요청이 한 줄의 구조화 로그가 된다 (docs/01 §5).

**측정값 원본과 질문 원문은 로그에 넣지 않는다.** `profile_ref` 는 불투명 참조라
로그만으로 사람을 특정할 수 없다 (docs/03 §2.4).

핸들러가 줄에 얹을 값은 `add_fields` 로 넣는다. 요청 하나가 끝나면 사라진다.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any

_fields: ContextVar[dict[str, Any] | None] = ContextVar("log_fields", default=None)

logger = logging.getLogger("family_fitness_ai.request")

# 로그에 넣지 않기로 한 것. 실수로 얹히면 여기서 걸린다.
FORBIDDEN_FIELDS = frozenset(
    {"measurements", "question", "answer", "height_cm", "weight_kg", "birth_date", "email"}
)


def reset_fields() -> None:
    _fields.set({})


def add_fields(**kwargs: Any) -> None:
    """이 요청의 로그 줄에 값을 얹는다."""
    forbidden = FORBIDDEN_FIELDS & kwargs.keys()
    if forbidden:
        raise ValueError(f"로그에 넣지 않기로 한 값이다 (docs/01 §5): {sorted(forbidden)}")
    current = _fields.get()
    if current is None:
        current = {}
        _fields.set(current)
    current.update(kwargs)


def current_fields() -> dict[str, Any]:
    return dict(_fields.get() or {})


def emit(**base: Any) -> None:
    """줄 하나를 낸다. 얹힌 값이 뒤에 붙는다."""
    logger.info(json.dumps({**base, **current_fields()}, ensure_ascii=False, sort_keys=True))
