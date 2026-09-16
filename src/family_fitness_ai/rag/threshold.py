"""`SIM_THRESHOLD` 를 잰다 (docs/dev/AI-8 §5 · docs/04 §3).

**임의의 숫자를 하드코딩하지 않는다.** 근거 없는 임계값은 조용히 관련 자료를 버린다.

재는 방법은 이렇다.

1. 질의 세트(`data/release/sim_queries.csv`)는 질의마다 **찾아야 할 청크를 미리 적어 둔다**.
   사람이 나중에 관련 여부를 고르는 것이 아니라, 질의를 만들 때 정답을 함께 정한다 —
   같은 세트로 다시 재야 모델을 바꿨을 때 비교가 된다
2. 질의마다 연령 필터를 걸고 후보를 찾는다 (검색과 같은 경로다)
3. 정답과 정답이 아닌 후보의 점수 분포에서 경계를 잡는다
4. 임계값 후보마다 정밀도·재현율을 내고, 고른 값과 이유를 남긴다

`무관` 질의는 정답이 없다 — 임계값을 넘는 청크가 하나도 없어야 한다.

실행 (rag.index 뒤, 임베딩 서버를 띄운 채):
    python -m family_fitness_ai.rag.threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.settings import INDEX_DIR, RELEASE_DIR
from . import chunks as K
from .embed import Embedder, embedder_from_settings
from .search import Corpus

QUERIES_FILE = "sim_queries.csv"
EVAL_FILE = "sim_eval.csv"
# 임계값 후보. 0.01 눈금이면 고른 값이 경계에 얼마나 붙어 있는지 보인다.
CANDIDATES = np.round(np.arange(0.30, 0.81, 0.01), 2)
# 후보를 넉넉히 본다 — 임계값을 정하기 전이라 top-k 로 자르면 분포가 잘린다.
EVAL_TOP_K = 50


def evaluate(queries: pd.DataFrame, corpus: Corpus, embedder: Embedder) -> pd.DataFrame:
    """질의 × 후보 청크 × 점수 × 정답 여부. 한 행이 판정 하나다."""
    vectors = embedder.embed(queries["query"].tolist())
    rows: list[dict[str, object]] = []
    for vector, query in zip(vectors, queries.to_dict("records"), strict=True):
        expected = {c for c in str(query["expected_chunk_ids"]).split(";") if c}
        # 임계값 0 으로 찾아 분포를 그대로 본다 — 자르는 것은 뒤에서 한다
        result = corpus.search(
            vector, [str(query["age_group"])], threshold=0.0, top_k=EVAL_TOP_K, limit=EVAL_TOP_K
        )
        for hit in result.hits:
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "kind": query["kind"],
                    "chunk_id": hit.chunk_id,
                    "source": hit.source,
                    "score": round(hit.score, 4),
                    "expected": hit.chunk_id in expected,
                }
            )
        for chunk_id in sorted(expected - {h.chunk_id for h in result.hits}):
            # 후보에도 못 든 정답 — 임계값과 무관하게 못 찾는다. 0 점으로 남긴다
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "kind": query["kind"],
                    "chunk_id": chunk_id,
                    "source": chunk_id.split(":")[0],
                    "score": 0.0,
                    "expected": True,
                }
            )
    return pd.DataFrame(rows)


def scores(evaluation: pd.DataFrame) -> pd.DataFrame:
    """임계값 후보마다 정밀도·재현율과, 무관 질의에서 새는 청크 수."""
    rows = []
    unrelated = evaluation["kind"] == "무관"
    for threshold in CANDIDATES:
        kept = evaluation["score"] >= threshold
        true_positive = int((kept & evaluation["expected"]).sum())
        false_positive = int((kept & ~evaluation["expected"]).sum())
        expected_total = int(evaluation["expected"].sum())
        rows.append(
            {
                "threshold": threshold,
                "recall": round(true_positive / expected_total, 3) if expected_total else 0.0,
                "precision": round(true_positive / (true_positive + false_positive), 3)
                if true_positive + false_positive
                else 0.0,
                "found": true_positive,
                "extra": false_positive,
                # 무관 질의에서 살아남은 청크 — 0 이어야 한다
                "unrelated_kept": int((kept & unrelated).sum()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SIM_THRESHOLD 를 잰다")
    ap.add_argument("--release", default=str(RELEASE_DIR), help="질의 세트가 있는 곳")
    ap.add_argument("--index", default=str(INDEX_DIR), help="색인이 있는 곳")
    ap.add_argument("--interim", default="data/interim", help="판정 표를 낼 곳")
    args = ap.parse_args(argv)

    queries = K.read_csv(Path(args.release) / QUERIES_FILE)
    corpus = Corpus.load(Path(args.index))
    evaluation = evaluate(queries, corpus, embedder_from_settings())
    out = Path(args.interim) / EVAL_FILE
    evaluation.to_csv(out, index=False, encoding="utf-8-sig")

    table = scores(evaluation)
    print(f"질의 {len(queries)} · 판정 {len(evaluation):,} → {out}")
    print(table.to_string(index=False))
    found = evaluation[evaluation["expected"]]
    print(f"\n정답 점수: 최소 {found.score.min():.3f} · 중앙 {found.score.median():.3f}")
    others = evaluation[~evaluation["expected"]]
    print(f"정답 아닌 후보: 중앙 {others.score.median():.3f} · 최대 {others.score.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
