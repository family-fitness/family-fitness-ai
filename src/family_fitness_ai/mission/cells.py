"""처방 칸 표와 「어느 칸을 쓰나」 (docs/05).

미션 편성의 재료는 전부 이 표에서 나온다. 한 행이 `(연령대 · 나이 · 성별 · 단계)`
한 칸이고, 그 칸에서 처방된 운동 이름을 **전부** 빈도 내림차순으로 싣는다.

**`rag.prescription` 의 청크와 다른 점은 자르지 않는다는 것뿐이다.** 청크는 임베딩
토큰 한도 때문에 칸·단계당 30개로 잘린다 (`MAX_EXERCISES`). 편성은 그보다 많이
봐야 하므로 빈도표를 따로 낸다. 자연키·인용 문구·표본 30 미만 제외는 청크와 같은
함수를 쓴다 — 두 표의 `chunk_id` 가 문자 그대로 같아야 인용이 이어진다.

표 만들기는 원자료가 있어야 하고, 칸 찾기는 커밋된 CSV만 있으면 된다.

실행 (저장소 루트에서):
    python -m family_fitness_ai.mission.cells --data-dir <원자료 디렉터리>
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..common.settings import RELEASE_DIR
from ..common.types import AgeGroup, age_unit_of
from ..ingest import measurements as M
from ..rag.prescription import (
    COVERAGE,
    MAX_EXERCISES,
    MIN_ROWS,
    PHASES,
    SOURCE,
    Term,
    build_vocabulary,
    citation_label,
    identity,
    natural_key,
    parse,
)

CELLS_FILE = "prescription_cells.csv"

# 다중값 한 칸의 구분자 (docs/02 §4 ②).
MULTI = ";"

# 칸이 잡히는 나이 범위. 자료에 칸이 있는 범위이고, 유소년만 아래로 넓다.
#
# **유소년을 7세까지 넓힌 것은 측정이 없어서다.** 처방 자료의 유소년 나이는 11·12
# 뿐인데 (docs/02 §2.2) 앱 대상은 만 4~15세다. `resolve_age_group` 이 만 7~10세를
# 유소년으로 두므로 (docs/02 §6 ④) 여기서도 받고 가장 가까운 11 로 당긴다.
#
# **어르신을 위로 넓히지 않은 것은 당기면 인용이 거짓이 되기 때문이다.** 95세
# 이상은 표본 30 미만이라 칸이 없다. 만 100세를 94 로 당기면 인용 문구가
# 「어르신 94세」가 된다 — 없으면 없다고 낸다.
#
# 유아기는 **개월**이다 (docs/02 §2.4). 원자료의 유아기 `MESURE_AGE_CO` 가 48~83
# 이고 세 단위 값이 0건인 것을 확인했다.
COVERED_AGES: dict[AgeGroup, tuple[int, int]] = {
    "유아기": (48, 83),
    "유소년": (7, 12),
    "청소년": (13, 18),
    "성인": (19, 64),
    "어르신": (65, 94),
}


@dataclass(frozen=True)
class Cell:
    """칸·단계 한 행. `exercise_names` 와 `exercise_counts` 는 같은 순서다."""

    chunk_id: str
    citation_label: str
    age_group: str
    age: int
    age_unit: str
    sex: str
    phase: str
    n: int
    exercise_names: tuple[str, ...]
    exercise_counts: tuple[int, ...]


# --------------------------------------------------------------------------
# 표 만들기 — 원자료가 필요하다. 시험에서 부르지 않는다.
# --------------------------------------------------------------------------


def build_cells(df: pd.DataFrame, vocabulary: list[Term]) -> list[Cell]:
    """칸·단계마다 한 행. **자르지 않는다.**

    표본 `MIN_ROWS` 미만 칸은 청크와 같이 뺀다 — 청크에 없는 칸을 편성이 쓰면
    「고른 운동이 인용한 청크 안에 있는가」 검사를 통과할 수 없다.
    """
    display = {identity(t.name): t.name for t in vocabulary}
    counts: dict[tuple[str, int, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    rows: Counter[tuple[str, int, str]] = Counter()

    for age_group, age, sex, text in zip(
        df[M.AGE_GROUP_COL], df[M.AGE_COL], df[M.SEX_COL], df[M.PRESCRIPTION_COL], strict=True
    ):
        cell = (str(age_group), int(age), str(sex))
        rows[cell] += 1
        for phase, names in parse(text).items():
            for raw in names:
                counts[cell][phase][display[identity(raw)]] += 1

    out: list[Cell] = []
    for cell in sorted(counts):
        age_group, age, sex = cell
        n = rows[cell]
        if n < MIN_ROWS:
            continue
        for phase in PHASES:
            counter = counts[cell].get(phase)
            if not counter:
                continue
            # 청크와 같은 정렬이다 (`_cover` 도 `most_common`). 앞 30개가 청크에
            # 들어간 것들이라, 편성이 「청크 안에 있는가」를 자리로 알 수 있다.
            ranked = counter.most_common()
            out.append(
                Cell(
                    chunk_id=f"{SOURCE}:{natural_key(age_group, age, sex, phase)}",
                    citation_label=citation_label(age_group, age),
                    age_group=age_group,
                    age=age,
                    age_unit=age_unit_of(age_group),  # type: ignore[arg-type]
                    sex=sex,
                    phase=phase,
                    n=n,
                    exercise_names=tuple(name for name, _ in ranked),
                    exercise_counts=tuple(count for _, count in ranked),
                )
            )
    return out


def chunk_prefix(cell: Cell) -> int:
    """이 칸·단계의 이름 중 **몇 개까지가 청크에 들어갔나**.

    `rag.prescription._cover` 와 같은 계산이다 — 누적 `COVERAGE` 를 덮을 때까지,
    최대 `MAX_EXERCISES` 개. `exercise_names` 가 청크와 같은 정렬이라 「앞 k개」로
    답이 나온다. 편성이 고른 운동이 인용한 청크 안에 있는지를 이 값으로 가른다.

    원자료도 청크 파일도 필요 없다 — `exercise_counts` 만으로 계산된다.
    """
    total = sum(cell.exercise_counts)
    covered = 0
    k = 0
    for count in cell.exercise_counts:
        if k >= MAX_EXERCISES or covered / total >= COVERAGE:
            break
        covered += count
        k += 1
    return k


def cells_frame(cells: list[Cell]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": c.chunk_id,
                "citation_label": c.citation_label,
                "age_group": c.age_group,
                "age": c.age,
                "age_unit": c.age_unit,
                "sex": c.sex,
                "phase": c.phase,
                "n": c.n,
                "exercise_names": MULTI.join(c.exercise_names),
                "exercise_counts": MULTI.join(str(x) for x in c.exercise_counts),
            }
            for c in cells
        ]
    )


# --------------------------------------------------------------------------
# 칸 찾기 — 커밋된 CSV만 있으면 된다.
# --------------------------------------------------------------------------


def load_cells(path: str | Path = RELEASE_DIR / CELLS_FILE) -> list[Cell]:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"exercise_counts": str})
    return [
        Cell(
            chunk_id=str(r.chunk_id),
            citation_label=str(r.citation_label),
            age_group=str(r.age_group),
            age=int(r.age),
            age_unit=str(r.age_unit),
            sex=str(r.sex),
            phase=str(r.phase),
            n=int(r.n),
            exercise_names=tuple(str(r.exercise_names).split(MULTI)),
            exercise_counts=tuple(int(x) for x in str(r.exercise_counts).split(MULTI)),
        )
        for r in df.itertuples(index=False)
    ]


@dataclass(frozen=True)
class CellMatch:
    """칸 찾기 결과. **비어 있어도 예외가 아니다** — 호출자가 인용 0 으로 거부한다."""

    cells: tuple[Cell, ...] = ()
    age_group: str | None = None
    age: int | None = None
    sex: str | None = None
    # 요청 나이에 칸이 없어 가장 가까운 나이로 당긴 경우. `retrieve` 단계 요약에 남긴다.
    pulled_from: int | None = None
    pulled_to: int | None = None
    empty_reason: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.cells)

    @property
    def pulled(self) -> bool:
        return self.pulled_from is not None

    def phase(self, phase: str) -> Cell | None:
        for c in self.cells:
            if c.phase == phase:
                return c
        return None


def _index(cells: list[Cell]) -> dict[tuple[str, int, str], list[Cell]]:
    out: dict[tuple[str, int, str], list[Cell]] = defaultdict(list)
    for c in cells:
        out[(c.age_group, c.age, c.sex)].append(c)
    return out


def find_cells(
    cells: list[Cell],
    age_group: str | None,
    age: int,
    age_unit: str,
    sex: str,
) -> CellMatch:
    """쓸 칸을 단계 순서(`PHASES`)로 돌려준다.

    규칙 —
    - **성별을 넓히지 않는다.** 그 성별 칸이 없으면 같은 나이의 다른 성별로 가지
      않는다. 넓히면 「또래 여아 처방」이라는 인용 문구가 거짓이 된다
    - 그 연령대 안에서 같은 성별의 **가장 가까운 나이**로 당긴다. 동거리면 아래쪽.
      당김이 일어나면 `pulled_from`·`pulled_to` 에 실린다
    - 나이가 `COVERED_AGES` 밖이면 빈 결과다. 예외를 던지지 않는다
    """
    if age_group is None:
        return CellMatch(empty_reason="연령대가 정해지지 않는다", sex=sex)
    if age_group not in COVERED_AGES:
        return CellMatch(empty_reason=f"모르는 연령대: {age_group}", sex=sex)

    expected_unit = age_unit_of(age_group)
    if age_unit != expected_unit:
        # 유아기 칸은 개월 단위다 (docs/02 §2.4). `세` 로 온 만 4~6세는 60~71개월
        # 처럼 두 해에 걸쳐 환산이 유일하지 않아 되돌리지 않는다 (docs/03 §3.1).
        return CellMatch(
            empty_reason=f"{age_group} 칸은 {expected_unit} 단위다 (받은 것: {age_unit})",
            age_group=age_group,
            sex=sex,
        )

    lo, hi = COVERED_AGES[age_group]
    if not lo <= age <= hi:
        return CellMatch(
            empty_reason=f"{age_group} 칸은 {lo}~{hi}{expected_unit} 뿐이다 (받은 것: {age})",
            age_group=age_group,
            sex=sex,
        )

    index = _index(cells)
    same_sex = sorted(a for (g, a, s) in index if g == age_group and s == sex)
    if not same_sex:
        return CellMatch(
            empty_reason=f"{age_group} {sex} 칸이 없다",
            age_group=age_group,
            sex=sex,
        )

    target = age if age in same_sex else min(same_sex, key=lambda a: (abs(a - age), a))
    found = index[(age_group, target, sex)]
    ordered = tuple(c for phase in PHASES for c in found if c.phase == phase)
    return CellMatch(
        cells=ordered,
        age_group=age_group,
        age=target,
        sex=sex,
        pulled_from=None if target == age else age,
        pulled_to=None if target == age else target,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="처방 칸 표를 CSV로 낸다")
    ap.add_argument("--data-dir", required=True, help="원자료 월별 CSV 디렉터리")
    ap.add_argument("--release", default=str(RELEASE_DIR), help="표를 낼 곳 (커밋한다)")
    args = ap.parse_args(argv)

    df = M.load_prescriptions(args.data_dir)
    cells = build_cells(df, build_vocabulary(df))

    release = Path(args.release)
    release.mkdir(parents=True, exist_ok=True)
    out = release / CELLS_FILE
    # utf-8-sig — 검수하는 사람이 엑셀로 연다 (docs/02 §4 ③)
    cells_frame(cells).to_csv(out, index=False, encoding="utf-8-sig")

    boxes = {(c.age_group, c.age, c.sex) for c in cells}
    in_chunk = sum(chunk_prefix(c) for c in cells)
    full = sum(len(c.exercise_names) for c in cells)
    print(f"처방 행 {len(df):,}")
    print(f"처방 칸 {len(boxes):,} · 칸×단계 {len(cells):,} · 후보 {in_chunk:,}")
    print(f"  └ 후보 {in_chunk:,} 는 청크에 들어간 것만이다. 이 표는 안 자른다 → {full:,}")
    for group in COVERED_AGES:
        rows = [c for c in cells if c.age_group == group]
        if not rows:
            continue
        cut = {n for c in rows for n in c.exercise_names[: chunk_prefix(c)]}
        names = {n for c in rows for n in c.exercise_names}
        boxes_here = len({(c.age, c.sex) for c in rows})
        top, all_of = sum(chunk_prefix(c) for c in rows), sum(len(c.exercise_names) for c in rows)
        print(
            f"  {group} 칸 {boxes_here} · 칸×단계 {len(rows)} "
            f"· 고유 운동 {len(cut)} → 안 자르면 {len(names)} "
            f"· 후보 {top:,} → {all_of:,}"
        )
    print(f"표 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
