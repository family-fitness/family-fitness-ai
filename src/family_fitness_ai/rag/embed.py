"""청크와 질의를 벡터로 바꾼다 (docs/04 §2.2).

`bge-m3` · **1024차원**. 로컬 `llama serve --embedding` 의 OpenAI 호환
`/v1/embeddings` 를 부른다 — 외부 호출도 키도 없다.

**모델을 바꾸면 전량 재색인이고 `SIM_THRESHOLD` 도 다시 재야 한다** (docs/04 §5).
그래서 개발과 배포가 같은 모델을 쓴다. 바꿔 끼울 수는 있게 두되, 바꾸는 순간 무엇이
따라오는지 여기 적어 둔다.

**벡터를 정규화해서 돌려준다.** 코사인 유사도를 내적 하나로 재기 위해서다 —
FAISS 와 pgvector 가 같은 값을 내야 `SIM_THRESHOLD` 가 한 벌로 선다.

띄우기:
    llama serve -hf gpustack/bge-m3-GGUF -hff bge-m3-Q8_0.gguf --embedding --port 8082
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..common.settings import get_settings

Json = dict[str, Any]
Post = Callable[[str, bytes], Json]

EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024
# 한 번에 보내는 청크 수. 서버가 슬롯마다 나눠 처리한다.
BATCH = 32


class EmbeddingFailed(RuntimeError):
    """서버가 없거나 모델이 다르다. 색인을 멈춘다 — 반쪽 벡터로 색인하지 않는다."""


class Embedder(Protocol):
    def version(self) -> str:
        """`embed:<model>/<dim>` (docs/01 §3.3)."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """`(len(texts), dim)` 정규화된 행렬."""
        ...

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        """청크 길이 검사에 쓴다 — 임베딩할 모델의 토크나이저로 세야 뜻이 있다."""
        ...


def _post_json(url: str, body: bytes) -> Json:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        payload: Json = json.load(response)
    return payload


@dataclass
class LlamaEmbedder:
    """로컬 `llama serve --embedding`."""

    url: str
    model: str = EMBEDDING_MODEL
    dim: int = EMBEDDING_DIM
    batch: int = BATCH
    post: Post = _post_json  # 시험에서 바꿔 끼운다

    def version(self) -> str:
        return f"embed:{self.model}/{self.dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            body = {"input": list(texts[start : start + self.batch]), "model": self.model}
            out = self.post(f"{self.url.rstrip('/')}/v1/embeddings", json.dumps(body).encode())
            # 서버가 순서를 바꿔 돌려줄 수 있다. `index` 로 되돌린다.
            returned = [item["embedding"] for item in sorted(out["data"], key=lambda d: d["index"])]
            # 묶음마다 센다. 전체 개수만 보면 한 묶음이 모자라고 다른 묶음이 남을 때
            # 합이 맞아 버려, 청크와 벡터가 한 칸씩 밀린 채로 색인된다.
            if len(returned) != len(body["input"]):
                raise EmbeddingFailed(
                    f"{start}번째 묶음에 {len(returned)}개가 왔다 — {len(body['input'])}개여야 한다"
                )
            rows += returned
        vectors = np.asarray(rows, dtype="float32")
        if vectors.shape != (len(texts), self.dim):
            raise EmbeddingFailed(f"{vectors.shape} — {len(texts)}×{self.dim} 이어야 한다")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        """`/tokenize` 로 센다. 임베딩할 모델과 같은 토크나이저다 (docs/04 §2.1)."""
        counts = []
        for text in texts:
            out = self.post(
                f"{self.url.rstrip('/')}/tokenize", json.dumps({"content": text}).encode()
            )
            counts.append(len(out["tokens"]))
        return counts


def embedder_from_settings() -> Embedder:
    return LlamaEmbedder(url=get_settings().embedding_url)
