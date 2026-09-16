"""임베딩 층 (docs/dev/AI-8 §1.3).

네트워크 없이 돈다 — 서버를 흉내 낸다. 모델이 한국어를 잘 가르는지는 여기서 재지 않는다.
그것은 `SIM_THRESHOLD` 측정에서 잰다 (docs/dev/AI-8 §5).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from family_fitness_ai.rag import embed as E


def server(dim: int = E.EMBEDDING_DIM, shuffle: bool = False) -> tuple[E.Post, list[list[str]]]:
    """받은 묶음을 기록하고, 글자 수로 만든 벡터를 돌려준다."""
    sent: list[list[str]] = []

    def post(url: str, body: bytes) -> dict[str, Any]:
        texts = json.loads(body)["input"]
        sent.append(texts)
        data = [
            {"index": i, "embedding": [float(len(t))] + [0.0] * (dim - 1)}
            for i, t in enumerate(texts)
        ]
        return {"data": list(reversed(data)) if shuffle else data}

    return post, sent


def test_정규화해서_돌려준다() -> None:
    """코사인 유사도를 내적 하나로 재기 위해서다 — 두 백엔드가 같은 값을 내야 한다."""
    post, _ = server()
    vectors = E.LlamaEmbedder(url="http://127.0.0.1:8082", post=post).embed(["가", "나다"])
    assert vectors.shape == (2, E.EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_묶어서_보내고_순서를_되돌린다() -> None:
    post, sent = server(shuffle=True)
    texts = [f"청크 {i}" * (i + 1) for i in range(5)]
    embedder = E.LlamaEmbedder(url="http://127.0.0.1:8082/", batch=2, post=post)

    vectors = embedder.embed(texts)

    assert [len(batch) for batch in sent] == [2, 2, 1]
    # 서버가 순서를 바꿔 돌려줘도 `index` 로 제자리에 놓는다
    assert [round(float(v[0]), 3) for v in vectors] == [1.0] * 5
    assert embedder.version() == "embed:bge-m3/1024"


def test_차원이_다르면_멈춘다() -> None:
    """반쪽 벡터로 색인하지 않는다 — 모델이 바뀌면 전량 재색인이다 (docs/04 §5)."""
    post, _ = server(dim=768)
    with pytest.raises(E.EmbeddingFailed):
        E.LlamaEmbedder(url="http://127.0.0.1:8082", post=post).embed(["가"])


def test_빈_입력은_서버를_부르지_않는다() -> None:
    post, sent = server()
    vectors = E.LlamaEmbedder(url="http://127.0.0.1:8082", post=post).embed([])
    assert (vectors.shape, sent) == ((0, E.EMBEDDING_DIM), [])
