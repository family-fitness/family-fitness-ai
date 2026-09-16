"""청크를 벡터로 바꿔 색인한다 (docs/04 · docs/04 §5).

**배치만 색인한다.** 온라인 요청을 처리하는 중에 코퍼스가 바뀌는 일은 없다
(docs/01 §4.2).

색인 전에 검사하고 **통과하지 못하면 색인하지 않는다** (docs/04).
빠뜨린 것은 세어서 보고한다 — 조용히 거르면 코퍼스가 왜 작은지 알 수 없다.

실행 (rag.chunks 뒤, 임베딩 서버를 띄운 채):
    python -m family_fitness_ai.rag.index --corpus-version 2026-09-16.1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import pandas as pd

from ..common.settings import INDEX_DIR, RELEASE_DIR
from . import chunks as K
from .embed import Embedder, embedder_from_settings

INDEX_FILE = "corpus.faiss"
META_FILE = "corpus_meta.csv"
MANIFEST_FILE = "corpus.json"

# docs/04 §2.1 — 너무 짧은 청크는 검색 잡음이고, 너무 긴 청크는 인용이 흐려진다.
MIN_TOKENS = 50
MAX_TOKENS = 400
# 연령이 반드시 있어야 하는 소스 (docs/04 §2.2). `criteria` 는 연령 무관이다.
AGED_SOURCES = ("prescription", "video")


class IndexRefused(RuntimeError):
    """색인 전 검사를 통과하지 못했다. 고치기 전에는 색인하지 않는다."""


def check(frame: pd.DataFrame, token_counts: list[int]) -> list[str]:
    """어긋난 것을 사람이 읽을 문장으로 돌려준다. 빈 목록이면 색인해도 된다."""
    problems: list[str] = []

    duplicated = frame["chunk_id"][frame["chunk_id"].duplicated()].tolist()
    if duplicated:
        problems.append(f"chunk_id 가 겹친다 {len(duplicated)}개 — 예: {duplicated[:3]}")

    aged = frame["source"].isin(AGED_SOURCES)
    missing_age = frame.loc[aged & (frame["age_group"] == ""), "chunk_id"].tolist()
    if missing_age:
        problems.append(f"연령이 빈 청크 {len(missing_age)}개 — 예: {missing_age[:3]}")

    tokens = pd.Series(token_counts, index=frame.index)
    short = frame.loc[tokens < MIN_TOKENS, "chunk_id"].tolist()
    long = frame.loc[tokens > MAX_TOKENS, "chunk_id"].tolist()
    if short:
        problems.append(f"{MIN_TOKENS}토큰 미만 {len(short)}개 — 예: {short[:3]}")
    if long:
        problems.append(f"{MAX_TOKENS}토큰 초과 {len(long)}개 — 예: {long[:3]}")
    return problems


def build(frame: pd.DataFrame, embedder: Embedder, corpus_version: str) -> tuple[Any, dict]:
    """검사를 통과하면 벡터를 만들고 색인을 돌려준다."""
    texts = frame["text"].tolist()
    problems = check(frame, embedder.token_counts(texts))
    if problems:
        raise IndexRefused(" / ".join(problems))

    vectors = embedder.embed(texts)
    # 정규화된 벡터의 내적 = 코사인 유사도 (embed.py)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    manifest = {
        "corpus_version": corpus_version,
        "embedding_version": embedder.version(),
        "dim": int(vectors.shape[1]),
        "chunks": int(len(frame)),
        "sources": frame["source"].value_counts().sort_index().to_dict(),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return index, manifest


def save(index: Any, frame: pd.DataFrame, manifest: dict, out_dir: Path) -> None:
    """색인·메타데이터·판 정보를 함께 쓴다. 셋은 늘 같이 움직인다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))
    # 행 순서가 벡터 순서다 — 검색이 이 순서로 되돌린다
    frame.to_csv(out_dir / META_FILE, index=False, encoding="utf-8-sig")
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="청크를 벡터로 색인한다")
    ap.add_argument("--corpus-version", required=True, help="코퍼스를 다시 만든 판 (docs/04 §5)")
    ap.add_argument("--release", default=str(RELEASE_DIR), help="chunks.csv 가 있는 곳")
    ap.add_argument("--out", default=str(INDEX_DIR), help="색인을 낼 곳 (커밋하지 않는다)")
    args = ap.parse_args(argv)

    frame = K.read_csv(Path(args.release) / K.CHUNKS_FILE)
    try:
        index, manifest = build(frame, embedder_from_settings(), args.corpus_version)
    except IndexRefused as e:
        print(f"[중단] 색인 전 검사를 통과하지 못했다 — {e}")
        return 1

    out_dir = Path(args.out)
    save(index, frame, manifest, out_dir)
    print(f"색인 {manifest['chunks']:,}청크 · {manifest['dim']}차원 → {out_dir}")
    print(f"  {manifest['embedding_version']} · corpus {manifest['corpus_version']}")
    for source, count in manifest["sources"].items():
        print(f"  {source} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
