"""임계값 후보 채점 (docs/dev/AI-8 §5).

임베딩 서버 없이 돈다 — 판정 표를 직접 만든다.
"""

from __future__ import annotations

import pandas as pd

from family_fitness_ai.rag import threshold as T


def judgements(*rows: tuple[str, str, float, bool]) -> pd.DataFrame:
    """(query_id, kind, score, expected)."""
    return pd.DataFrame(
        [
            {"query_id": q, "kind": kind, "chunk_id": f"c{i}", "score": score, "expected": expected}
            for i, (q, kind, score, expected) in enumerate(rows)
        ]
    )


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
    assert (high["recall"], high["found"], high["extra"]) == (0.5, 1, 0)


def test_무관_질의에서_살아남는_청크를_따로_센다() -> None:
    """무관 질의는 정답이 없다 — 임계값을 넘는 청크가 하나도 없어야 한다."""
    table = T.scores(judgements(("q1", "처방", 0.70, True), ("q9", "무관", 0.48, False)))
    assert at(table, 0.45)["unrelated_kept"] == 1
    assert at(table, 0.50)["unrelated_kept"] == 0


def test_후보에도_못_든_정답은_0점으로_재현율에_반영된다() -> None:
    table = T.scores(judgements(("q1", "처방", 0.70, True), ("q2", "영상", 0.0, True)))
    assert at(table, 0.50)["recall"] == 0.5
