"""코퍼스 인덱스를 읽는 쪽."""

from __future__ import annotations

import csv

import pytest
from conftest import needs_index

from family_fitness_ai.rag import index

#: Windows 의 C long 은 64비트 파이썬에서도 32비트다. 3.11 은 이보다 큰 값을
#: csv.field_size_limit 에 넘기면 OverflowError 를 낸다.
_C_LONG_MAX = 2**31 - 1


@pytest.fixture
def windows_csv(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Windows · 파이썬 3.11 의 csv.field_size_limit 흉내. 넘긴 값을 적어 둔다."""
    asked: list[int] = []
    real = csv.field_size_limit

    def field_size_limit(*args: int) -> int:
        if args:
            if args[0] > _C_LONG_MAX:
                raise OverflowError("Python int too large to convert to C long")
            asked.append(args[0])
        return real(*args)

    monkeypatch.setattr(csv, "field_size_limit", field_size_limit)
    return asked


def test_long_fields_are_allowed_within_a_c_long(windows_csv):
    index.allow_long_fields()
    assert windows_csv and all(value <= _C_LONG_MAX for value in windows_csv)


def test_factors_split_on_either_separator():
    # 처방 줄은 「·」 로, 영상 줄은 「;」 로 요인을 잇는다.
    # 한쪽만 나누면 영상 요인이 한 덩어리가 된다.
    assert index._factors("근력;근지구력;협응력") == ("근력", "근지구력", "협응력")
    assert index._factors("근력·유연성") == ("근력", "유연성")
    assert index._factors("") == ()


@needs_index
def test_video_chunks_keep_each_factor_apart():
    index.corpus.cache_clear()
    try:
        joined = [f for c in index.corpus().chunks for f in c.factors if ";" in f or "·" in f]
        assert joined == []
    finally:
        index.corpus.cache_clear()


@needs_index
def test_the_corpus_loads_where_a_c_long_is_32_bits(windows_csv):
    index.corpus.cache_clear()
    try:
        assert index.corpus().chunks
    finally:
        index.corpus.cache_clear()
