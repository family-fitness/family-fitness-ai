"""영상 라벨 LLM 층 (docs/02).

네트워크 없이 돈다 — 로컬 서버와 Anthropic SDK 를 흉내 낸다. 모델이 잘 보는지는 여기서
재지 않는다. 그것은 시험 묶음에서 잰다.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types.beta import BetaTextBlock

from family_fitness_ai.common.settings import get_settings
from family_fitness_ai.labeling import llm

SCHEMA = {
    "type": "object",
    "properties": {"jumping": {"type": "boolean"}},
    "required": ["jumping"],
    "additionalProperties": False,
}
JPEG = b"\xff\xd8\xff not really a jpeg"


@pytest.fixture(autouse=True)
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── 로컬 ────────────────────────────────────────────────────────────


def test_로컬_서버에는_이미지를_data_URL_로_보내고_스키마로_묶는다() -> None:
    sent: dict[str, Any] = {}

    def post(url: str, body: bytes) -> dict[str, Any]:
        sent["url"], sent["body"] = url, json.loads(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"jumping": true}'}}]}

    labeler = llm.LlamaServer(url="http://127.0.0.1:8081/", model="qwen3-vl-8b", post=post)

    assert labeler.observe([JPEG, JPEG], "보이는 것만 답한다", SCHEMA) == {"jumping": True}
    assert sent["url"] == "http://127.0.0.1:8081/v1/chat/completions"
    content = sent["body"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "보이는 것만 답한다"}
    assert (
        content[1]["image_url"]["url"]
        == "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()
    )
    assert len(content) == 3
    assert sent["body"]["response_format"]["json_schema"]["schema"] == SCHEMA
    assert labeler.version() == "llama:qwen3-vl-8b"


def test_잘린_답은_라벨로_쓰지_않는다() -> None:
    def post(url: str, body: bytes) -> dict[str, Any]:
        return {"choices": [{"finish_reason": "length", "message": {"content": '{"jump'}}]}

    with pytest.raises(llm.LabelerRefused):
        llm.LlamaServer(url="http://127.0.0.1:8081", post=post).observe([JPEG], "p", SCHEMA)


# ── Claude ──────────────────────────────────────────────────────────


class FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def claude(stop_reason: str, text: str = "") -> tuple[llm.ClaudeLabeler, FakeMessages]:
    response = SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="bio") if stop_reason == "refusal" else None,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            BetaTextBlock(type="text", text=text),
        ],
    )
    messages = FakeMessages(response)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return llm.ClaudeLabeler(client=client), messages


def test_Claude_에는_이미지를_지시보다_앞에_두고_스키마와_대체_모델을_건다() -> None:
    labeler, messages = claude("end_turn", '{"jumping": false}')

    assert labeler.observe([JPEG], "보이는 것만 답한다", SCHEMA) == {"jumping": False}

    sent = messages.kwargs
    assert sent["model"] == "claude-opus-5"
    assert (sent["fallbacks"], sent["betas"]) == ("default", ["server-side-fallback-2026-07-01"])
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    content = sent["messages"][0]["content"]
    assert content[0]["source"] == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": base64.standard_b64encode(JPEG).decode(),
    }
    assert content[-1] == {"type": "text", "text": "보이는 것만 답한다"}


def test_Claude_가_거절하면_라벨을_비우도록_올린다() -> None:
    labeler, _ = claude("refusal")
    with pytest.raises(llm.LabelerRefused, match="거절"):
        labeler.observe([JPEG], "p", SCHEMA)


# ── 바꿔 끼우기 ─────────────────────────────────────────────────────


def test_백엔드는_환경변수_하나로_바뀐다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LABELER_BACKEND", "claude")
    assert isinstance(llm.labeler_from_settings(), llm.ClaudeLabeler)

    get_settings.cache_clear()
    monkeypatch.setenv("LABELER_BACKEND", "llama")
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:9000")
    labeler = llm.labeler_from_settings()
    assert isinstance(labeler, llm.LlamaServer)
    assert labeler.url == "http://127.0.0.1:9000"


def test_기본은_외부_호출이_없는_로컬이다(monkeypatch: pytest.MonkeyPatch) -> None:
    """외부 API 호출량이 늘어나는 변경은 합의가 먼저다 (AGENTS.md §3)."""
    monkeypatch.delenv("LABELER_BACKEND", raising=False)
    assert isinstance(llm.labeler_from_settings(), llm.LlamaServer)
