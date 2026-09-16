"""`SIM_THRESHOLD` 를 잰다 (docs/05 §4.2 · docs/04 §3).

**임의의 숫자를 하드코딩하지 않는다.** 근거 없는 임계값은 조용히 관련 자료를 버린다.

재는 방법은 이렇다.

1. 질의 세트(`data/release/sim_queries.csv`)는 질의마다 **찾아야 할 청크를 미리 적어 둔다**.
   사람이 나중에 관련 여부를 고르는 것이 아니라, 질의를 만들 때 정답을 함께 정한다 —
   같은 세트로 다시 재야 모델을 바꿨을 때 비교가 된다
2. 질의마다 연령 필터를 걸고 후보를 찾는다 (검색과 같은 경로다)
3. 정답과 정답이 아닌 후보의 점수 분포에서 경계를 잡는다
4. 임계값 후보마다 정밀도·재현율을 내고, 고른 값과 이유를 남긴다

**재현율은 실제 검색 경로로 잰다.** 후보를 `EVAL_TOP_K` 만큼 넓게 뽑는 것은 분포를
보기 위해서지, 그 풀 전체가 답변에 들어가서가 아니다. 코치가 실제로 보는 것은
`search.TOP_K` 안에서 임계값을 넘은 앞 `search.MAX_CONTEXT` 개다 (docs/04 §3).
풀 기준으로 재면 재현율이 부풀어 — 그 차이를 `recall_pool` 로 함께 낸다.

`무관` 질의는 정답이 없다 — **컨텍스트에 아무것도 들어가지 않아야 한다.**

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
from .search import MAX_CONTEXT, TOP_K, Corpus

QUERIES_FILE = "sim_queries.csv"
EVAL_FILE = "sim_eval.csv"
# 임계값 후보. 0.01 눈금이면 고른 값이 경계에 얼마나 붙어 있는지 보인다.
CANDIDATES = np.round(np.arange(0.30, 0.81, 0.01), 2)
# 후보를 넉넉히 본다 — 임계값을 정하기 전이라 top-k 로 자르면 분포가 잘린다.
# **이 값은 분포를 보기 위한 것이지 재현율의 기준이 아니다** (모듈 설명).
EVAL_TOP_K = 50
# 후보에도 못 든 정답을 나타내는 순위. 어떤 임계값에서도 컨텍스트에 들지 못한다.
UNRANKED = 10**6


def evaluate(queries: pd.DataFrame, corpus: Corpus, embedder: Embedder) -> pd.DataFrame:
    """질의 × 후보 청크 × 점수 × 순위 × 정답 여부. 한 행이 판정 하나다.

    `rank` 는 연령 필터를 거친 뒤의 순위다(0부터). 이 열이 있어야 임계값을 바꿔
    가며 **실제 경로**(`TOP_K` → `MAX_CONTEXT`)를 다시 질의 없이 재현할 수 있다.
    """
    vectors = embedder.embed(queries["query"].tolist())
    rows: list[dict[str, object]] = []
    for vector, query in zip(vectors, queries.to_dict("records"), strict=True):
        expected = {c for c in str(query["expected_chunk_ids"]).split(";") if c}
        # 임계값 0 으로 찾아 분포를 그대로 본다 — 자르는 것은 뒤에서 한다
        result = corpus.search(
            vector, [str(query["age_group"])], threshold=0.0, top_k=EVAL_TOP_K, limit=EVAL_TOP_K
        )
        for rank, hit in enumerate(result.hits):
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "kind": query["kind"],
                    "chunk_id": hit.chunk_id,
                    "source": hit.source,
                    "rank": rank,
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
                    "rank": UNRANKED,
                    "score": 0.0,
                    "expected": True,
                }
            )
    return pd.DataFrame(rows)


def in_context(
    evaluation: pd.DataFrame, threshold: float, top_k: int = TOP_K, limit: int = MAX_CONTEXT
) -> pd.Series:
    """실제 경로에서 컨텍스트에 들어가는 행 (`search.Corpus.search` 와 같은 순서다).

    `TOP_K` 안에서 임계값을 넘은 것을 앞에서부터 `MAX_CONTEXT` 개 — 순위는 이미
    연령 필터를 거친 뒤의 것이라 여기서 연령을 다시 볼 필요가 없다.
    """
    passing = (evaluation["rank"] < top_k) & (evaluation["score"] >= threshold)
    order = evaluation["rank"].where(passing, UNRANKED)
    place = order.groupby(evaluation["query_id"]).rank(method="first") - 1
    return passing & (place < limit)


def scores(evaluation: pd.DataFrame) -> pd.DataFrame:
    """임계값 후보마다의 재현율·정밀도와, 무관 질의에서 새는 양.

    `recall` 은 실제 경로다. `recall_pool` 은 후보 풀(`EVAL_TOP_K`) 전체를 답으로
    칠 때의 값 — 둘의 차이가 "넓게 뽑아서 잘 찾은 것처럼 보이는" 몫이다.
    """
    rows = []
    unrelated = evaluation["kind"] == "무관"
    unrelated_queries = evaluation.loc[unrelated, "query_id"].nunique()
    expected_total = int(evaluation["expected"].sum())
    for threshold in CANDIDATES:
        context = in_context(evaluation, threshold)
        pool = evaluation["score"] >= threshold
        true_positive = int((context & evaluation["expected"]).sum())
        false_positive = int((context & ~evaluation["expected"]).sum())
        leaked = evaluation.loc[context & unrelated, "query_id"].nunique()
        rows.append(
            {
                "threshold": threshold,
                "recall": round(true_positive / expected_total, 3) if expected_total else 0.0,
                "precision": round(true_positive / (true_positive + false_positive), 3)
                if true_positive + false_positive
                else 0.0,
                "found": true_positive,
                # 후보 풀 전체를 답으로 칠 때의 재현율 — 실제 경로와의 차이를 보려고 둔다
                "recall_pool": round(int((pool & evaluation["expected"]).sum()) / expected_total, 3)
                if expected_total
                else 0.0,
                # 무관 질의에서 컨텍스트에 청크가 든 질의 수 — 0 이어야 한다
                "unrelated_kept": int(leaked),
                "unrelated_total": int(unrelated_queries),
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
    print(f"재현율은 실제 경로다 — top_k={TOP_K} 안에서 임계값을 넘은 앞 {MAX_CONTEXT}개")
    print(table.to_string(index=False))

    gold = evaluation[evaluation["expected"]]
    reached = gold[gold["rank"] < UNRANKED]
    print(
        f"\n정답 {len(gold)}개 중 후보에 든 것 {len(reached)}개"
        f" — 점수 최소 {reached.score.min():.3f} · 중앙 {reached.score.median():.3f}"
    )
    unrelated = evaluation[evaluation["kind"] == "무관"]
    if not unrelated.empty:
        worst = unrelated.loc[unrelated["score"].idxmax()]
        print(
            f"무관 질의 최고 점수 {worst.score:.3f}"
            f" — {worst.query_id} `{worst['query']}` × {worst.chunk_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
