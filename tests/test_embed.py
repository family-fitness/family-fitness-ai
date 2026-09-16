"""임베딩 층 (docs/04 §2.2).

네트워크 없이 돈다 — 서버를 흉내 낸다. 모델이 한국어를 잘 가르는지는 여기서 재지 않는다.
그것은 `SIM_THRESHOLD` 측정에서 잰다 (docs/05 §4.2).
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


def test_묶음_하나가_모자라면_멈춘다() -> None:
    """전체 개수만 보면 한 묶음이 모자라고 다른 묶음이 남을 때 합이 맞아 버린다.

    그러면 청크와 벡터가 한 칸씩 밀린 채 색인돼, 검색이 엉뚱한 청크를 인용한다.
    """

    def post(url: str, body: bytes) -> dict[str, Any]:
        texts = json.loads(body)["input"]
        # 첫 묶음은 하나 모자라게, 다음 묶음은 하나 더 — 합은 맞는다
        count = len(texts) - 1 if len(sent) == 0 else len(texts) + 1
        sent.append(texts)
        return {
            "data": [
                {"index": i, "embedding": [1.0] + [0.0] * (E.EMBEDDING_DIM - 1)}
                for i in range(count)
            ]
        }

    sent: list[list[str]] = []
    embedder = E.LlamaEmbedder(url="http://x", batch=2, post=post)
    with pytest.raises(E.EmbeddingFailed, match="묶음"):
        embedder.embed(["가", "나", "다", "라"])


def test_토큰을_모델의_토크나이저로_센다() -> None:
    """청크 길이 경계(50~400)를 임베딩할 모델과 같은 잣대로 잰다 (docs/04 §2.1)."""
    asked: list[str] = []

    def post(url: str, body: bytes) -> dict[str, Any]:
        asked.append(url)
        return {"tokens": list(range(len(json.loads(body)["content"])))}

    embedder = E.LlamaEmbedder(url="http://x/", post=post)
    assert embedder.token_counts(["가나다", "가나"]) == [3, 2]
    assert asked == ["http://x/tokenize", "http://x/tokenize"]
