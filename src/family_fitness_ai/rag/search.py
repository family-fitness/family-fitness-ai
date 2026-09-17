"""코퍼스 검색. 연령 필터가 먼저고 유사도가 나중이다."""

from __future__ import annotations

from dataclasses import dataclass

from family_fitness_ai.common.settings import settings
from family_fitness_ai.rag.embed import embed_one
from family_fitness_ai.rag.index import Chunk, corpus


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Result:
    hits: list[Hit]
    #: 무엇 때문에 몇 건이 떨어졌나. 빈 결과가 결함이 아님을 보이는 자리다.
    filtered_out: dict[str, int]


def search(
    query: str,
    *,
    k: int = 5,
    sources: tuple[str, ...] = (),
    age_group: str | None = None,
    factors: tuple[str, ...] = (),
    threshold: float | None = None,
) -> Result:
    """질의에 가까운 청크를 연령·요인으로 거른 뒤 k 개까지 낸다.

    코퍼스가 4천여 줄이라 전부 훑고 나서 거른다. 먼저 자르고 거르면 연령이 맞는
    청크가 잘려 나가 빈 결과가 되는 일이 생긴다.
    """
    limit = threshold if threshold is not None else settings().sim_threshold
    vector = embed_one(query)
    store = corpus()
    scored = store.search(vector, len(store.chunks))

    dropped = {"age_group": 0, "below_threshold": 0, "source": 0, "factor": 0}
    hits: list[Hit] = []
    for chunk, score in scored:
        if sources and chunk.source not in sources:
            dropped["source"] += 1
            continue
        # 연령 라벨이 없는 청크는 아이 앞에 내지 않는다. 라벨이 없다는 것은
        # 그 영상이 누구 것인지 모른다는 뜻이다.
        if age_group and chunk.age_group != age_group:
            dropped["age_group"] += 1
            continue
        if factors and not (set(factors) & set(chunk.factors)):
            dropped["factor"] += 1
            continue
        if score < limit:
            dropped["below_threshold"] += 1
            continue
        hits.append(Hit(chunk=chunk, score=round(score, 4)))

    # 세다 말고 자르지 않는다. 몇 건이 왜 떨어졌는지가 「빈 결과는 결함이
    # 아니다」를 보이는 근거라, k 를 채운 뒤에도 끝까지 센다.
    return Result(hits=hits[:k], filtered_out={key: n for key, n in dropped.items() if n})
