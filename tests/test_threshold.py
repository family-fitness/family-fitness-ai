"""임계값 후보 채점 (docs/05 §4.2).

임베딩 서버 없이 돈다 — 판정 표를 직접 만든다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from family_fitness_ai.rag import threshold as T


def judgements(*rows: tuple[str, str, float, bool]) -> pd.DataFrame:
    """(query_id, kind, score, expected). 순위는 질의마다 점수 내림차순으로 매긴다."""
    frame = pd.DataFrame(
        [
            {"query_id": q, "kind": kind, "chunk_id": f"c{i}", "score": score, "expected": expected}
            for i, (q, kind, score, expected) in enumerate(rows)
        ]
    )
    rank = frame.groupby("query_id")["score"].rank(method="first", ascending=False) - 1
    # 0점은 후보에도 못 든 정답이다 (`evaluate` 가 그렇게 남긴다)
    frame["rank"] = rank.astype(int).where(frame["score"] > 0, T.UNRANKED)
    return frame


def at(table: pd.DataFrame, threshold: float) -> dict[str, float]:
    return table[table.threshold == threshold].iloc[0].to_dict()


def test_임계값을_올리면_재현율이_떨어지고_정밀도가_오른다() -> None:
    table = T.scores(
        judgements(
            ("q1", "처방", 0.70, True),
            ("q1", "처방", 0.55, True),
            ("q1", "처방", 0.60, False),
        )
    )
    low, high = at(table, 0.50), at(table, 0.65)
    assert (low["recall"], low["precision"]) == (1.0, 0.667)
    assert (high["recall"], high["found"]) == (0.5, 1)


def test_무관_질의는_컨텍스트가_비어야_한다() -> None:
    """무관 질의는 정답이 없다 — 하나라도 들어가면 그 질의를 센다."""
    table = T.scores(judgements(("q1", "처방", 0.70, True), ("q9", "무관", 0.48, False)))
    assert at(table, 0.45)["unrelated_kept"] == 1
    assert at(table, 0.50)["unrelated_kept"] == 0
    assert at(table, 0.50)["unrelated_total"] == 1


def test_무관_질의는_청크가_아니라_질의_수로_센다() -> None:
    """한 질의가 청크 셋을 통과시켜도 새는 질의는 하나다 — 16개 중 몇 개인지가 뜻이다."""
    table = T.scores(
        judgements(
            ("q8", "무관", 0.70, False),
            ("q8", "무관", 0.68, False),
            ("q8", "무관", 0.66, False),
            ("q9", "무관", 0.40, False),
        )
    )
    assert at(table, 0.50)["unrelated_kept"] == 1
    assert at(table, 0.50)["unrelated_total"] == 2


def test_후보에도_못_든_정답은_재현율을_깎는다() -> None:
    table = T.scores(judgements(("q1", "처방", 0.70, True), ("q2", "영상", 0.0, True)))
    assert at(table, 0.50)["recall"] == 0.5


def test_재현율은_컨텍스트_상한까지만_센다() -> None:
    """실제 경로는 top_k 안에서 임계값을 넘은 앞 MAX_CONTEXT 개다 (docs/04 §3).

    풀 전체를 답으로 치면 재현율이 부풀어 — 그 차이가 `recall_pool` 이다.
    """
    over = T.MAX_CONTEXT + 2
    table = T.scores(judgements(*[("q1", "처방", 0.90 - i * 0.01, True) for i in range(over)]))
    row = at(table, 0.50)
    assert row["found"] == T.MAX_CONTEXT
    assert row["recall"] == round(T.MAX_CONTEXT / over, 3)
    assert row["recall_pool"] == 1.0


def test_top_k_밖의_정답은_점수가_높아도_들어가지_않는다() -> None:
    """임계값을 넘어도 1차 top-k 밖이면 컨텍스트에 못 든다."""
    rows = [("q1", "처방", 0.90 - i * 0.001, False) for i in range(T.TOP_K)]
    rows.append(("q1", "처방", 0.80, True))  # 순위는 TOP_K 번째 — 창 밖이다
    table = T.scores(judgements(*rows))
    assert at(table, 0.50)["recall"] == 0.0
    assert at(table, 0.50)["recall_pool"] == 1.0


class FakeCorpus:
    """`Corpus.search` 자리를 대신한다 — 색인도 서버도 없이 `evaluate` 를 잰다."""

    def __init__(self, hits: dict[str, list[tuple[str, float]]]) -> None:
        self.hits = hits
        self.asked: list[list[str]] = []

    def search(self, vector, age_groups, threshold, top_k, limit):  # type: ignore[no-untyped-def]
        self.asked.append(list(age_groups))
        found = self.hits[str(vector[0])]
        return type(
            "R",
            (),
            {
                "hits": [
                    type(
                        "H",
                        (),
                        {"chunk_id": c, "source": c.split(":")[0], "score": s},
                    )()
                    for c, s in found[:limit]
                ]
            },
        )()


class FakeEmbedder:
    def version(self) -> str:
        return "embed:fake/2"

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return np.array([[float(i), 0.0] for i in range(len(texts))], dtype="float32")

    def token_counts(self, texts):  # type: ignore[no-untyped-def]
        return [100] * len(texts)


def test_evaluate_는_찾은_순서를_순위로_남긴다() -> None:
    queries = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query": "다섯 살 아이 운동",
                "age_group": "유아기",
                "kind": "처방",
                "expected_chunk_ids": "prescription:a;prescription:없는것",
            }
        ]
    )
    corpus = FakeCorpus({"0.0": [("prescription:b", 0.8), ("prescription:a", 0.7)]})
    out = T.evaluate(queries, corpus, FakeEmbedder())

    assert corpus.asked == [["유아기"]], "연령 필터를 걸고 물어야 한다"
    found = out.set_index("chunk_id")
    assert found.loc["prescription:b", "rank"] == 0
    assert found.loc["prescription:a", "rank"] == 1
    assert bool(found.loc["prescription:a", "expected"]) is True
    # 후보에 들지 못한 정답도 행으로 남는다 — 조용히 빠지면 재현율이 부푼다
    assert found.loc["prescription:없는것", "rank"] == T.UNRANKED
    assert found.loc["prescription:없는것", "score"] == 0.0
