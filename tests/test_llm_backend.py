"""모델을 바꿔도 부름이 어긋나지 않나.

여기서도 LLM 을 부르지 않는다. .env 를 고치는 손이 하는 일 — 모델 이름만 바꾸기,
백엔드만 바꾸기, 둘을 서로 다르게 적기 — 을 넣어 보고 어디로 가는지만 본다.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass

import pytest

from family_fitness_ai.common import llm

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


@dataclass
class FakeSettings:
    coach_backend: str = "claude"
    coach_model: str = ""
    anthropic_api_key: str = "sk-test"
    gemini_api_key: str = "g-test"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch):
    """어느 백엔드에 어떤 모델로 갔는지만 적어 둔다."""
    seen: list[tuple[str, str]] = []

    def claude(prompt, schema, system, model, max_tokens, timeout):
        seen.append(("claude", model))
        return {"ok": True}

    def gemini(prompt, schema, system, model, max_tokens):
        seen.append(("gemini", model))
        return {"ok": True}

    monkeypatch.setattr(llm, "_claude", claude)
    monkeypatch.setattr(llm, "_gemini", gemini)
    monkeypatch.setattr(llm, "_resting", {})
    return seen


def _env(monkeypatch: pytest.MonkeyPatch, **kwargs) -> None:
    config = FakeSettings(**kwargs)
    monkeypatch.setattr(llm, "settings", lambda: config)


def test_model_name_tells_us_the_backend():
    assert llm.backend_for("claude-haiku-4-5", "gemini") == "claude"
    assert llm.backend_for("gemini-flash-latest", "claude") == "gemini"
    # 모르는 이름이면 적혀 있는 백엔드를 그대로 쓴다.
    assert llm.backend_for("", "claude") == "claude"
    assert llm.backend_for("my-model", "gemini") == "gemini"


def test_changing_only_the_model_moves_the_backend(monkeypatch, calls):
    _env(monkeypatch, coach_backend="claude", coach_model="gemini-flash-latest")
    llm.ask_json("q", SCHEMA, "s")
    assert calls[0] == ("gemini", "gemini-flash-latest")


def test_changing_only_the_backend_picks_its_own_model(monkeypatch, calls):
    _env(monkeypatch, coach_backend="gemini", coach_model="")
    llm.ask_json("q", SCHEMA, "s")
    assert calls[0] == ("gemini", llm.GEMINI_DEFAULT)


def test_an_unknown_backend_name_does_not_stop_us(monkeypatch, calls):
    _env(monkeypatch, coach_backend="openai", coach_model="")
    llm.ask_json("q", SCHEMA, "s")
    assert calls[0] == ("claude", llm.CLAUDE_DEFAULT)


def test_crossing_over_does_not_carry_the_other_backends_model(monkeypatch, calls):
    """클로드 모델 이름을 들고 제미나이로 넘어가면 그쪽에서 400 이 난다."""
    _env(monkeypatch, coach_backend="claude", coach_model="claude-haiku-4-5")
    monkeypatch.setattr(llm, "_claude", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("503")))
    monkeypatch.setattr(llm, "COOLDOWN_SEC", 0)
    llm.ask_json("q", SCHEMA, "s", timeout=0.1)
    assert calls[-1] == ("gemini", llm.GEMINI_DEFAULT)


def test_a_backend_without_a_key_is_skipped(monkeypatch, calls):
    _env(monkeypatch, coach_backend="gemini", coach_model="", gemini_api_key="")
    llm.ask_json("q", SCHEMA, "s")
    assert [which for which, _ in calls] == ["claude"]


def test_we_read_json_even_through_a_fence():
    assert llm.loads('{"ok": true}') == {"ok": True}
    assert llm.loads('```json\n{"ok": true}\n```') == {"ok": True}
    assert llm.loads('여기 있습니다: {"ok": true} 확인해 보세요') == {"ok": True}
    with pytest.raises(json.JSONDecodeError):
        llm.loads("드릴 수 있는 것이 없습니다")


class _Block:
    type = "text"
    text = '{"ok": true}'


class _Reply:
    content = [_Block()]


def _fake_anthropic(monkeypatch: pytest.MonkeyPatch, *, refuses_effort: bool) -> list[dict]:
    """effort 를 거절하는 모델 흉내. 보낸 output_config 를 적어 둔다."""
    sent: list[dict] = []

    class BadRequestError(Exception):
        pass

    class Messages:
        def create(self, **kwargs):
            sent.append(kwargs["output_config"])
            if refuses_effort and "effort" in kwargs["output_config"]:
                raise BadRequestError("output_config.effort: unsupported for this model")
            return _Reply()

    class Anthropic:
        def __init__(self, **kwargs):
            self.messages = Messages()

    module = types.ModuleType("anthropic")
    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    module.BadRequestError = BadRequestError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setattr(llm, "_NO_EFFORT", set())
    return sent


def test_a_model_that_refuses_effort_still_answers(monkeypatch):
    _env(monkeypatch)
    sent = _fake_anthropic(monkeypatch, refuses_effort=True)
    assert llm._claude("q", SCHEMA, "s", "claude-haiku-4-5", 100, 5.0) == {"ok": True}
    assert "effort" in sent[0] and "effort" not in sent[1]
    # 한 번 거절당한 모델에는 다시 보내지 않는다 — 부를 때마다 두 번 가면 안 된다.
    sent.clear()
    llm._claude("q", SCHEMA, "s", "claude-haiku-4-5", 100, 5.0)
    assert len(sent) == 1 and "effort" not in sent[0]


def test_a_model_that_takes_effort_gets_it(monkeypatch):
    _env(monkeypatch)
    sent = _fake_anthropic(monkeypatch, refuses_effort=False)
    llm._claude("q", SCHEMA, "s", "claude-opus-5", 100, 5.0)
    assert len(sent) == 1 and sent[0]["effort"] == "low"


def _fake_genai(monkeypatch: pytest.MonkeyPatch, *, refuses_quick: bool) -> list[dict]:
    """생각을 끌 수 없는 모델 흉내. 보낸 config 를 적어 둔다."""
    import google

    sent: list[dict] = []

    class Models:
        def generate_content(self, *, model, contents, config):
            sent.append(config)
            if refuses_quick and "thinking_config" in config:
                raise ValueError("thinking_budget must be greater than 0 for this model")
            return types.SimpleNamespace(text='{"ok": true}')

    class Client:
        def __init__(self, **kwargs):
            self.models = Models()

    module = types.ModuleType("genai")
    module.Client = Client  # type: ignore[attr-defined]
    monkeypatch.setattr(google, "genai", module, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", module)
    monkeypatch.setattr(llm, "_NO_EFFORT", set())
    return sent


def test_gemini_does_not_spend_the_answer_budget_on_thinking(monkeypatch):
    """생각도 max_output_tokens 에서 깎인다 — 켜 두면 빈 글이 돌아온 적이 있다."""
    _env(monkeypatch)
    sent = _fake_genai(monkeypatch, refuses_quick=False)
    assert llm._gemini("q", SCHEMA, "s", "gemini-flash-latest", 1500) == {"ok": True}
    assert sent[0]["thinking_config"] == {"thinking_budget": 0}


def test_a_gemini_model_that_must_think_still_answers(monkeypatch):
    _env(monkeypatch)
    sent = _fake_genai(monkeypatch, refuses_quick=True)
    assert llm._gemini("q", SCHEMA, "s", "gemini-2.5-pro", 1500) == {"ok": True}
    assert "thinking_config" in sent[0] and "thinking_config" not in sent[1]
