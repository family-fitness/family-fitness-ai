"""처방 칸 표와 칸 찾기 (docs/05).

**원자료를 읽지 않는다.** 커밋된 `data/release/prescription_cells.csv` 만 본다 —
원자료 284MB 는 CI 에 없다. 표를 만드는 쪽(`build_cells`·`main`)은 부르지 않는다.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd
import pytest

from family_fitness_ai.common.types import resolve_age_group
from family_fitness_ai.mission import cells as C
from family_fitness_ai.rag.prescription import PHASES, SOURCE, natural_key

RELEASE = Path(__file__).resolve().parents[1] / "data" / "release"
CELLS_CSV = RELEASE / C.CELLS_FILE
VOCABULARY_CSV = RELEASE / "exercise_vocabulary.csv"

# 앱 대상은 만 4~15세다. 만 4~6세는 개월로 온다 (docs/03 §3.1).
CHILD_YEARS = range(7, 16)
TODDLER_MONTHS = range(48, 84)


@pytest.fixture(scope="module")
def cells() -> list[C.Cell]:
    return C.load_cells(CELLS_CSV)


@pytest.fixture(scope="module")
def vocabulary() -> set[str]:
    frame = pd.read_csv(VOCABULARY_CSV, encoding="utf-8-sig")
    return set(frame["exercise_name"])


def find(cells: list[C.Cell], age: int, unit: str, sex: str) -> C.CellMatch:
    """계약이 하는 것과 같은 순서 — 연령대를 먼저 정하고 칸을 찾는다."""
    return C.find_cells(cells, resolve_age_group(age, unit), age, unit, sex)  # type: ignore[arg-type]


# --- 표 --------------------------------------------------------------------


def test_칸과_단계마다_한_행씩_708행이다(cells: list[C.Cell]) -> None:
    assert len(cells) == 708
    boxes = {(c.age_group, c.age, c.sex) for c in cells}
    assert len(boxes) == 236
    # 236 × 3 = 708 — 단계가 빠진 칸이 없다
    assert {c.phase for c in cells} == set(PHASES)
    assert all(len([c for c in cells if (c.age_group, c.age, c.sex) == b]) == 3 for b in boxes)


def test_chunk_id_가_청크의_자연키와_문자_그대로_같다(cells: list[C.Cell]) -> None:
    for c in cells:
        assert c.chunk_id == f"{SOURCE}:{natural_key(c.age_group, c.age, c.sex, c.phase)}"
    assert len({c.chunk_id for c in cells}) == len(cells)


def test_인용_문구가_708행_전부_비지_않는다(cells: list[C.Cell]) -> None:
    assert all(c.citation_label.strip() for c in cells)
    assert all(c.citation_label.startswith("국민체력100 운동처방 · ") for c in cells)
    labels = {c.citation_label for c in cells}
    assert "국민체력100 운동처방 · 유소년 11세" in labels
    # 유아기는 개월이다 (docs/02 §2.4)
    assert "국민체력100 운동처방 · 유아기 48개월" in labels


def test_이름과_횟수가_같은_길이이고_빈도_내림차순이다(cells: list[C.Cell]) -> None:
    for c in cells:
        assert len(c.exercise_names) == len(c.exercise_counts) > 0
        counts = list(c.exercise_counts)
        assert counts == sorted(counts, reverse=True)


def test_운동_이름이_전부_처방_어휘_안에_있다(cells: list[C.Cell], vocabulary: set[str]) -> None:
    assert len(vocabulary) == 641
    outside = {n for c in cells for n in c.exercise_names} - vocabulary
    assert outside == set()


def test_청크에_들어간_상위_고유_운동_수가_실측과_맞는다(cells: list[C.Cell]) -> None:
    """실측값은 **청크(상위 30)** 기준이다. 이 표는 안 자르므로 수가 더 크다."""
    expected = {"유아기": 129, "유소년": 62, "청소년": 102, "성인": 102, "어르신": 181}
    uncut = {"유아기": 466, "유소년": 325, "청소년": 425, "성인": 508, "어르신": 466}
    for group, want in expected.items():
        rows = [c for c in cells if c.age_group == group]
        top = {n for c in rows for n in c.exercise_names[: C.chunk_prefix(c)]}
        assert len(top) == want, group
        assert len({n for c in rows for n in c.exercise_names}) == uncut[group], group


def test_자르지_않았다는_증거로_30을_넘는_칸이_있다(cells: list[C.Cell]) -> None:
    sizes = [len(c.exercise_names) for c in cells]
    assert min(sizes) == 12
    assert statistics.median(sizes) == 109
    assert max(sizes) == 342
    assert sum(1 for s in sizes if s > 30) == 685
    # 청크 쪽은 30 을 넘지 않는다 — 자르는 경계가 표 안에서 읽힌다
    assert max(C.chunk_prefix(c) for c in cells) == 30
    assert sum(C.chunk_prefix(c) for c in cells) == 15_525


# --- 칸 찾기 ----------------------------------------------------------------


def test_만_7세부터_15세까지_모든_나이에서_칸이_잡힌다(cells: list[C.Cell]) -> None:
    for age in CHILD_YEARS:
        for sex in ("M", "F"):
            match = find(cells, age, "세", sex)
            assert match.found, (age, sex)
            assert [c.phase for c in match.cells] == list(PHASES)
            assert all(c.sex == sex for c in match.cells)


def test_만_7에서_10세는_11로_당겨지고_당김이_실린다(cells: list[C.Cell]) -> None:
    for age in (7, 8, 9, 10):
        match = find(cells, age, "세", "F")
        assert match.pulled
        assert (match.pulled_from, match.pulled_to) == (age, 11)
        assert match.age_group == "유소년"
        assert all(c.age == 11 for c in match.cells)
        assert all("유소년 11세" in c.citation_label for c in match.cells)
    # 칸이 있는 나이는 당기지 않는다
    for age in (11, 12, 13, 15):
        assert not find(cells, age, "세", "F").pulled


def test_48에서_83개월_모든_개월에서_칸이_잡힌다(cells: list[C.Cell]) -> None:
    for month in TODDLER_MONTHS:
        for sex in ("M", "F"):
            match = find(cells, month, "개월", sex)
            assert match.found, (month, sex)
            assert not match.pulled
            assert match.age_group == "유아기"
            assert all(c.age_unit == "개월" for c in match.cells)


def test_만_4에서_6세를_세로_주면_빈_결과이고_이유가_실린다(cells: list[C.Cell]) -> None:
    """유아기는 개월로 받는다 (docs/03 §3.1) — 만 5세가 60~71개월이라 환산이 하나가 아니다."""
    for age in (4, 5, 6):
        match = find(cells, age, "세", "M")
        assert not match.found
        assert match.empty_reason
    # 연령대를 유아기로 못박아 단위만 틀리게 줘도 마찬가지다
    mismatch = C.find_cells(cells, "유아기", 5, "세", "M")
    assert not mismatch.found
    assert "개월" in (mismatch.empty_reason or "")


def test_어르신_65에서_94세도_칸이_잡힌다(cells: list[C.Cell]) -> None:
    for age in range(65, 95):
        for sex in ("M", "F"):
            match = find(cells, age, "세", sex)
            assert match.found, (age, sex)
            assert match.age_group == "어르신"
    # 남자 칸은 90세까지다 (91~94는 표본 30 미만) → 90 으로 당긴다
    for age in (91, 92, 93, 94):
        pulled = find(cells, age, "세", "M")
        assert (pulled.pulled_from, pulled.pulled_to) == (age, 90)
        assert not find(cells, age, "세", "F").pulled


def test_성별_칸이_없으면_다른_성별로_넓히지_않는다(cells: list[C.Cell]) -> None:
    """넓히면 「또래 여아 처방」이라는 인용 문구가 거짓이 된다."""
    for age in range(65, 95):
        for sex in ("M", "F"):
            assert all(c.sex == sex for c in find(cells, age, "세", sex).cells)
    only_men = [c for c in cells if c.age_group == "어르신" and c.sex == "M"]
    assert max(c.age for c in only_men) == 90


def test_칸_범위_밖은_예외가_아니라_빈_결과다(cells: list[C.Cell]) -> None:
    for age in (3, 100):
        match = find(cells, age, "세", "M")
        assert not match.found
        assert match.cells == ()
        assert match.empty_reason
    # 만 3세는 연령대 자체가 정해지지 않는다. 만 100세는 어르신이지만 칸이 없다
    assert resolve_age_group(3, "세") is None
    assert find(cells, 100, "세", "M").age_group == "어르신"
    # 개월도 같다
    assert not find(cells, 47, "개월", "F").found
    assert not find(cells, 84, "개월", "F").found


def test_같은_입력이_같은_결과를_낸다(cells: list[C.Cell]) -> None:
    for age, unit, sex in ((8, "세", "F"), (60, "개월", "M"), (94, "세", "M"), (40, "세", "F")):
        first = find(cells, age, unit, sex)
        assert first == find(cells, age, unit, sex)
        # 표를 다시 읽어도 같다
        assert first == find(C.load_cells(CELLS_CSV), age, unit, sex)
