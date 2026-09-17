"""질의를 벡터로 바꾼다.

인덱스를 만들 때 쓴 임베딩과 같은 모델이어야 한다 — 다르면 점수가 뜻을 잃는다.
모델 이름은 data/index/corpus.json 의 embedding_version 에 적혀 있다.
"""

from __future__ import annotations

import httpx
import numpy as np

from family_fitness_ai.common.errors import temporarily_unavailable
from family_fitness_ai.common.settings import settings

_TIMEOUT = 5.0


def embed(texts: list[str]) -> np.ndarray:
    """(n, dim) 단위벡터. 인덱스가 내적이라 정규화해야 코사인이 된다."""
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    url = settings().embedding_url.rstrip("/") + "/v1/embeddings"
    try:
        response = httpx.post(url, json={"input": texts}, timeout=_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise temporarily_unavailable("임베딩 서버에 닿지 못했습니다") from exc

    rows = [item["embedding"] for item in payload["data"]]
    vectors = np.asarray(rows, dtype="float32")
    if vectors.ndim == 3:  # llama.cpp 가 토큰별로 낼 때가 있다
        vectors = vectors[:, 0, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
