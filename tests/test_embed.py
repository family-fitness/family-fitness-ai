"""임베딩을 어디서 돌리나.

여기서는 진짜 모델을 올리지 않는다. llama_cpp 자리에 흉내만 넣고, 고른 길로
가는지·한 번만 올리는지·끝날 때 내리는지를 본다. 진짜 모델과 인덱스가 맞는지는
`python -m family_fitness_ai.rag.embed` 가 잰다.
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from family_fitness_ai.common.settings import settings
from family_fitness_ai.rag import embed


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch):
    """모델을 올리지 않은 채로 시작한다."""
    monkeypatch.setattr(embed, "_model", None)
    monkeypatch.setattr(settings(), "embedding_gguf", "")
    return monkeypatch


def _fake_llama(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Llama 흉내. 몇 번 올렸고 몇 번 닫았는지 센다."""
    counts = {"loaded": 0, "closed": 0}

    class Llama:
        def __init__(self, **kwargs):
            counts["loaded"] += 1

        def embed(self, texts, truncate=True):
            return [[3.0, 4.0] for _ in texts]

        def close(self):
            counts["closed"] += 1

    module = types.ModuleType("llama_cpp")
    module.Llama = Llama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setattr(embed, "model_path", lambda **_: embed.Path("bge-m3-Q8_0.gguf"))
    monkeypatch.setattr(embed.atexit, "register", lambda fn: None)
    return counts


@pytest.mark.parametrize(
    ("written", "chosen"),
    [("", "local"), ("local", "local"), ("http", "http"), (" HTTP ", "http"), ("gpu", "local")],
)
def test_backend_is_local_unless_http_is_written(fresh, written, chosen):
    fresh.setattr(settings(), "embedding_backend", written)
    assert embed.backend() == chosen


def test_http_backend_asks_the_server(fresh):
    fresh.setattr(settings(), "embedding_backend", "http")
    fresh.setattr(embed, "_over_http", lambda texts: [[0.0, 2.0]] * len(texts))
    fresh.setattr(embed, "_in_process", lambda texts: pytest.fail("local 로 가면 안 된다"))
    assert embed.embed_one("질의").tolist() == [0.0, 1.0]


def test_local_backend_loads_once_and_normalizes(fresh):
    fresh.setattr(settings(), "embedding_backend", "local")
    counts = _fake_llama(fresh)
    first = embed.embed(["하나", "둘"])
    embed.embed_one("셋")
    assert counts["loaded"] == 1
    assert np.allclose(first, [[0.6, 0.8], [0.6, 0.8]])


def test_threads_share_one_model(fresh):
    """편성 스레드와 요청 스레드가 한꺼번에 불러도 모델은 하나만 올린다."""
    fresh.setattr(settings(), "embedding_backend", "local")
    counts = _fake_llama(fresh)
    threads = [threading.Thread(target=embed.embed_one, args=("질의",)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counts["loaded"] == 1


def test_close_releases_the_model(fresh):
    """끝나기 전에 닫지 않으면 Metal 쪽이 abort 한다(종료 코드 134)."""
    fresh.setattr(settings(), "embedding_backend", "local")
    counts = _fake_llama(fresh)
    embed.embed_one("질의")
    embed._close()
    embed._close()  # 두 번 불러도 한 번만 닫는다
    assert counts["closed"] == 1
    assert embed._model is None


def test_missing_says_what_to_install(fresh):
    fresh.setattr(settings(), "embedding_backend", "local")
    fresh.setattr(embed, "find_spec", lambda name: None)
    assert "pip install -e '.[embed]'" in embed.missing()


def test_missing_says_how_to_get_the_model(fresh):
    fresh.setattr(settings(), "embedding_backend", "local")
    fresh.setattr(embed, "find_spec", lambda name: object())

    def not_cached(**_):
        raise FileNotFoundError("허깅페이스 캐시에 없다")

    fresh.setattr(embed, "model_path", not_cached)
    assert "make embed-model" in embed.missing()


def test_missing_points_at_the_written_path(fresh, tmp_path):
    """적어 둔 자리가 비었으면 make embed-model 을 권하지 않는다 — 그건 캐시에 받는다."""
    fresh.setattr(settings(), "embedding_backend", "local")
    fresh.setattr(settings(), "embedding_gguf", str(tmp_path / "없는.gguf"))
    fresh.setattr(embed, "find_spec", lambda name: object())
    reason = embed.missing()
    assert "EMBEDDING_GGUF" in reason
    assert "make embed-model" not in reason


def test_missing_is_empty_for_http(fresh):
    """밖의 서버가 잠깐 내려간 것은 띄울 때 막지 않는다 — 요청마다 503 이면 된다."""
    fresh.setattr(settings(), "embedding_backend", "http")
    fresh.setattr(embed, "find_spec", lambda name: None)
    assert embed.missing() == ""


def test_warm_up_loads_the_model_now(fresh):
    """띄울 때 올려 두면 첫 검색이 모델을 기다리지 않는다."""
    fresh.setattr(settings(), "embedding_backend", "local")
    counts = _fake_llama(fresh)
    embed.warm_up()
    assert counts["loaded"] == 1
    embed.embed_one("체력")
    assert counts["loaded"] == 1


def test_warm_up_leaves_an_http_server_alone(fresh):
    """밖의 서버는 저쪽이 이미 올려 두었다. 띄울 때 그 서버에 기대지 않는다."""
    fresh.setattr(settings(), "embedding_backend", "http")
    asked: list[object] = []
    fresh.setattr(embed.httpx, "post", lambda *a, **k: asked.append(a))
    embed.warm_up()
    assert asked == []
