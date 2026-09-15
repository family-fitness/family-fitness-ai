"""영상 라벨용 LLM — 로컬(llama.cpp)과 Claude 를 바꿔 끼운다 (docs/dev/AI-7 §3.8).

**LLM 은 판단하지 않는다** (AGENTS.md §7). 화면에 보이는 것을 고정 목록에서 고르게 하고,
그것을 라벨로 바꾸는 것은 코드가 한다. 그래서 이 층은 한 가지만 한다 —
"JPEG 몇 장 + 지시 + JSON 스키마 → 스키마를 지킨 JSON".

`LABELER_BACKEND=llama|claude` 하나로 바뀌고 호출부는 같다. `VECTOR_BACKEND` 와 같은
방식이다 (docs/01 §1).

- `llama`: 로컬 `llama serve` 의 OpenAI 호환 API. 외부 호출이 없다
- `claude`: Anthropic API. **외부 호출이라 비용이 든다** (AGENTS.md §3)
"""

from __future__ import annotations

import base64
import json
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
from anthropic.types.beta import (
    BetaImageBlockParam,
    BetaMessageParam,
    BetaOutputConfigParam,
    BetaTextBlock,
    BetaTextBlockParam,
)

from ..common.settings import get_settings

Json = dict[str, Any]
Post = Callable[[str, bytes], Json]

DEFAULT_CLAUDE_MODEL = "claude-opus-5"
# Claude 는 생각하는 데도 이 한도를 쓴다. 답은 짧아도 넉넉히 둔다.
CLAUDE_MAX_TOKENS = 16000
# 거절되면 서버가 거절 사유에 맞는 모델로 다시 돌린다 (`fallbacks: "default"`).
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LabelerRefused(RuntimeError):
    """모델이 답하지 않았거나 답이 잘렸다. 그 칸은 비운다 — 지어내지 않는다."""


class VisionLabeler(Protocol):
    def version(self) -> str:
        """`labeler:<model>/<prompt-v>` 의 model 자리 (docs/01 §3.3)."""
        ...

    def observe(self, images: Sequence[bytes], prompt: str, schema: Json) -> Json:
        """JPEG 여러 장과 지시를 보내고 스키마를 지킨 JSON 을 받는다."""
        ...


def _b64(image: bytes) -> str:
    return base64.standard_b64encode(image).decode("ascii")


def _post_json(url: str, body: bytes) -> Json:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        payload: Json = json.load(response)
    return payload


@dataclass
class LlamaServer:
    """로컬 `llama serve` 의 `/v1/chat/completions`."""

    url: str
    model: str = "local"
    max_tokens: int = 512
    post: Post = _post_json  # 시험에서 바꿔 끼운다

    def version(self) -> str:
        return f"llama:{self.model}"

    def observe(self, images: Sequence[bytes], prompt: str, schema: Json) -> Json:
        content: list[Json] = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image)}"}}
            for image in images
        ]
        body = {
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,  # 같은 화면이면 같은 답
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "observation", "schema": schema},
            },
        }
        out = self.post(f"{self.url.rstrip('/')}/v1/chat/completions", json.dumps(body).encode())
        choice = out["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LabelerRefused("응답이 잘렸다")
        result: Json = json.loads(choice["message"]["content"])
        return result


@dataclass
class ClaudeLabeler:
    """Anthropic API. 자격증명은 환경에서 읽는다 (`ANTHROPIC_API_KEY` 등)."""

    model: str = DEFAULT_CLAUDE_MODEL
    client: anthropic.Anthropic | None = None  # 비우면 부를 때 만든다. 시험에서 바꿔 끼운다

    def version(self) -> str:
        return self.model

    def observe(self, images: Sequence[bytes], prompt: str, schema: Json) -> Json:
        client = self.client or anthropic.Anthropic()
        content: list[BetaImageBlockParam | BetaTextBlockParam] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(image)},
            }
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        messages: list[BetaMessageParam] = [{"role": "user", "content": content}]
        output_config: BetaOutputConfigParam = {"format": {"type": "json_schema", "schema": schema}}
        response = client.beta.messages.create(
            model=self.model,
            max_tokens=CLAUDE_MAX_TOKENS,
            betas=[CLAUDE_FALLBACK_BETA],
            fallbacks="default",
            output_config=output_config,
            messages=messages,
        )
        # 거절은 내용보다 먼저 본다. 대체 모델까지 거절했다는 뜻이다.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LabelerRefused(f"거절 · {category}")
        if response.stop_reason == "max_tokens":
            raise LabelerRefused("응답이 잘렸다")
        text = next(block.text for block in response.content if isinstance(block, BetaTextBlock))
        result: Json = json.loads(text)
        return result


def labeler_from_settings() -> VisionLabeler:
    """`LABELER_BACKEND` 하나로 바꿔 끼운다. 호출부는 같다."""
    settings = get_settings()
    if settings.labeler_backend == "claude":
        return ClaudeLabeler(model=settings.labeler_model or DEFAULT_CLAUDE_MODEL)
    return LlamaServer(url=settings.llama_server_url, model=settings.labeler_model or "local")
