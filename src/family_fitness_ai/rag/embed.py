"""질의를 벡터로 바꾼다.

인덱스를 만들 때 쓴 임베딩과 같은 모델이어야 한다 — 다르면 점수가 뜻을 잃는다.
모델 이름은 data/index/corpus.json 의 embedding_version 에 적혀 있다.

두 길이 있다 (.env 의 EMBEDDING_BACKEND).

    local  이 프로세스 안에서 돈다(기본). 인덱스를 만든 바로 그 파일
           bge-m3-Q8_0.gguf 를 llama-cpp-python 으로 읽는다. 인덱스에 든 벡터와
           재 보니 코사인 1.00000 이었다. 임베딩 서버를 따로 띄우지 않는다.
    http   밖에 띄운 llama.cpp 서버에 묻는다(EMBEDDING_URL).

local 은 띄울 때 모델을 올린다(warm_up, EMBEDDING_WARMUP). 처음 쓸 때 올리면 첫
검색이 모델을 기다리느라 10초 가까이 걸려 BE 의 읽기 한도(검색 4초)를 넘었다.
--reload 로 코드를 고칠 때마다 기다리기 싫으면 EMBEDDING_WARMUP=0 으로 끈다 —
그때는 처음 쓸 때 올린다.

    python -m family_fitness_ai.rag.embed    모델을 받아 두고 인덱스와 맞는지 잰다
"""

from __future__ import annotations

import atexit
import csv
import logging
import sys
import threading
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from family_fitness_ai.common.errors import temporarily_unavailable
from family_fitness_ai.common.settings import settings

log = logging.getLogger(__name__)

_TIMEOUT = 5.0

GGUF_REPO = "gpustack/bge-m3-GGUF"
GGUF_FILE = "bge-m3-Q8_0.gguf"
#: 한 번에 넣는 토큰 수. 질의는 수십 토큰이라 넉넉하고, 넘치면 자른다.
_CONTEXT = 2048

#: llama-cpp-python 의 모델은 여러 스레드가 한꺼번에 부르면 안 된다. 편성은 뒤쪽
#: 스레드에서, 요청은 스레드 풀에서 돌므로 한 번에 하나씩 들여보낸다. 한 건이
#: 10ms 남짓이라 기다림은 느끼기 어렵다.
_lock = threading.Lock()
_model: Any = None


def backend() -> str:
    """local 또는 http. 비었거나 못 보던 값이면 local 이다."""
    return "http" if settings().embedding_backend.strip().lower() == "http" else "local"


def embed(texts: list[str]) -> np.ndarray:
    """(n, dim) 단위벡터. 인덱스가 내적이라 정규화해야 코사인이 된다."""
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    rows = _over_http(texts) if backend() == "http" else _in_process(texts)
    vectors = np.asarray(rows, dtype="float32")
    if vectors.ndim == 3:  # llama.cpp 가 토큰별로 낼 때가 있다
        vectors = vectors[:, 0, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def warm_up() -> None:
    """local 모델을 지금 올린다. http 는 건드리지 않는다 — 저쪽 서버가 이미 올려 두었다."""
    if backend() != "local":
        return
    started = time.perf_counter()
    embed_one("체력")
    log.info("임베딩 모델을 올려 두었다 (%.1f초)", time.perf_counter() - started)


def _over_http(texts: list[str]) -> list[Any]:
    url = settings().embedding_url.rstrip("/") + "/v1/embeddings"
    try:
        response = httpx.post(url, json={"input": texts}, timeout=_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise temporarily_unavailable("임베딩 서버에 닿지 못했습니다") from exc
    return [item["embedding"] for item in payload["data"]]


def _in_process(texts: list[str]) -> list[Any]:
    global _model
    with _lock:
        if _model is None:
            _model = _load()
            atexit.register(_close)
        return list(_model.embed(texts, truncate=True))


def _close() -> None:
    """프로세스가 끝나기 전에 모델을 내린다.

    맡겨 두면 인터프리터가 모듈을 다 치운 뒤에야 Llama.__del__ 이 돌아 닫지 못하고,
    Metal 쪽이 쓰던 자원이 남았다며 abort 한다(종료 코드 134). 서버를 끌 때마다,
    --reload 로 다시 띄울 때마다 그랬다.
    """
    global _model
    # 편성 스레드가 막 임베딩하는 중이면 끝나기를 잠깐 기다린다. 한 건이 10ms 남짓이다.
    if not _lock.acquire(timeout=5):
        return
    try:
        if _model is not None:
            _model.close()
            _model = None
    finally:
        _lock.release()


def _load() -> Any:
    try:
        from llama_cpp import Llama

        path = model_path()
    except (ImportError, FileNotFoundError) as exc:
        # 띄울 때 missing() 이 먼저 막는다. 여기까지 왔으면 그 사이 무엇이 치워진 것이다.
        log.error("임베딩 모델을 올리지 못했다: %s", exc)
        raise temporarily_unavailable("임베딩 모델을 올리지 못했습니다") from exc
    log.info("임베딩 모델을 올린다: %s", path.name)
    return Llama(
        model_path=str(path),
        embedding=True,
        n_ctx=_CONTEXT,
        n_batch=_CONTEXT,
        n_ubatch=_CONTEXT,
        n_gpu_layers=-1,  # 맥은 Metal 로, GPU 가 없으면 CPU 로 돈다
        verbose=False,
    )


def model_path(*, download: bool = False) -> Path:
    """모델 파일 자리. 없으면 FileNotFoundError 다 — download 면 받아 온다(605 MB)."""
    configured = settings().embedding_gguf.strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(GGUF_REPO, GGUF_FILE, local_files_only=not download))


def missing() -> str:
    """local 로 돌 수 없는 까닭. 빈 문자열이면 돌 수 있다.

    모델을 올리지는 않고 있는지만 본다. http 는 여기서 보지 않는다 — 서버가 잠깐
    내려가 있는 것은 요청마다 503 으로 알리면 되는 일이다.
    """
    if backend() == "http":
        return ""
    if find_spec("llama_cpp") is None:
        return "llama-cpp-python 이 없다 — pip install -e '.[embed]'"
    if not settings().embedding_gguf.strip() and find_spec("huggingface_hub") is None:
        return "huggingface-hub 이 없다 — pip install -e '.[embed]'"
    try:
        model_path()
    except FileNotFoundError:
        configured = settings().embedding_gguf.strip()
        if configured:
            # make embed-model 은 허깅페이스 캐시에 받는다. 적어 둔 자리를 채우지 않는다.
            return f"EMBEDDING_GGUF 에 적은 파일이 없다 ({configured})"
        return f"모델 파일({GGUF_FILE})이 없다 — make embed-model"
    return ""


#: 인덱스와 견줄 청크 수와, 같은 모델로 칠 코사인.
SELF_CHECK = 12
SAME = 0.99


def main() -> None:
    """모델을 받아 두고, 인덱스에 든 벡터와 같게 나오는지 잰다."""
    if backend() == "http":
        print("EMBEDDING_BACKEND=http — 밖의 서버를 쓴다. 받을 모델이 없다.")
        return
    path = model_path(download=True)
    print(f"모델 {path.name} ({path.stat().st_size / 2**20:.0f} MB)")

    from family_fitness_ai.rag.index import allow_long_fields, corpus

    # 원문으로 견준다. corpus() 는 화면에 나갈 말로 고쳐 읽어서(「미달」→「참가」)
    # 인덱스를 만들 때의 글자와 다르다.
    allow_long_fields()
    meta = settings().index_dir / "corpus_meta.csv"
    with meta.open(encoding="utf-8-sig", newline="") as fh:
        texts = [row["text"] for row in csv.DictReader(fh)]
    picks = np.linspace(0, len(texts) - 1, SELF_CHECK, dtype=int)
    stored = np.stack([corpus().index.reconstruct(int(i)) for i in picks])
    cosine = (embed([texts[i] for i in picks]) * stored).sum(axis=1)

    print(f"인덱스와 견줌 {len(picks)}건 · 코사인 최소 {cosine.min():.5f}")
    if cosine.min() < SAME:
        print("인덱스를 만든 모델과 다르다 — 검색 점수가 뜻을 잃는다.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
