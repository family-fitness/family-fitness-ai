"""처방 어휘.

국민체력100 처방문에 실제로 적힌 운동 이름들이다. 영상 속 이름과 이어 붙이는
쪽의 기준말이 된다.
"""

from __future__ import annotations

import collections
import re
from functools import lru_cache

from family_fitness_ai.rag.index import corpus

#: 「달리기(9%)」 — 처방문은 운동 이름과 처방 비율을 이렇게 적는다.
EXERCISE = re.compile(r"(?P<name>[^,:]+?)\s*\((?P<pct>\d+)%\)")
_HEAD = "처방된 본운동:"


def exercises_in(text: str) -> list[tuple[str, int]]:
    """처방문 → [(운동명, 비율)]. 비율이 높은 것이 앞이다."""
    body = text.split(_HEAD, 1)
    if len(body) != 2:
        return []
    found = [
        (re.sub(r"\s+", " ", match["name"].strip()), int(match["pct"]))
        for match in EXERCISE.finditer(body[1])
    ]
    return sorted(found, key=lambda pair: -pair[1])


@lru_cache
def by_age_group() -> dict[str, tuple[str, ...]]:
    """연령대마다 그 또래에게 실제로 처방된 운동 이름."""
    out: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for chunk in corpus().chunks:
        if chunk.source != "prescription":
            continue
        for name, _ in exercises_in(chunk.text):
            out[chunk.age_group][name] += 1
    return {group: tuple(sorted(counts)) for group, counts in out.items()}


@lru_cache
def all_names() -> tuple[str, ...]:
    names: set[str] = set()
    for group in by_age_group().values():
        names |= set(group)
    return tuple(sorted(names))


def normalized(name: str) -> str:
    return re.sub(r"[\s·•\-_.]", "", name)
