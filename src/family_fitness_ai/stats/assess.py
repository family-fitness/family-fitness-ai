"""측정값 하나를 점수와 분포상 위치로 바꾼다.

원자료를 읽지 않는다. `data/release/` 의 산출물만으로 돈다 — 서비스가 기동 시
한 번 올려두고 요청마다 조회·보간만 하는 형태를 그대로 옮긴 것이다.

실행 (저장소 루트에서):
    python -m family_fitness_ai.stats.assess --age 11 --sex F \\
        --measure 028=52.3 --measure 012=12.6 --measure 020=66
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import criteria as C
from . import grade as G
from .items import ITEMS
from .score import Anchors, score

DEFAULT_RELEASE = Path("data/release")
BAND_STRENGTH, BAND_GROWTH = 75, 25
CRITERIA_FILE = "grade_thresholds.csv"
BODY_RANGES_FILE = "body_composition_ranges.csv"


@dataclass(frozen=True)
class Cell:
    """한 (연령대, 구간, 성별, 항목) 칸의 채점 재료."""

    age_group: str
    age_lo: int
    age_hi: int
    age_unit: str
    sex: str
    item_code: str
    n: int
    anchors: Anchors
    quantiles: np.ndarray  # 원값 p1~p99. 경험분포를 되살린다
    score_bins: np.ndarray  # 점수 10구간 비율


class Reference:
    """산출물 묶음. 기동 시 한 번 만들고 재사용한다."""

    def __init__(self, release_dir: str | Path = DEFAULT_RELEASE) -> None:
        root = Path(release_dir).expanduser()
        missing = [
            name
            for name in (
                "age_band_score_summary.csv",
                "age_band_score_distribution.csv",
                "age_band_value_quantiles.csv",
            )
            if not (root / name).exists()
        ]
        if missing:
            raise FileNotFoundError(f"산출물이 없다: {root} — {', '.join(missing)}")

        # 문턱은 등급 판정이 쓴다 (docs/dev/AI-2 §1). 없으면 점수만 낸다.
        criteria_path = root / CRITERIA_FILE
        self.thresholds: list[C.Threshold] = C.load(criteria_path) if criteria_path.exists() else []
        # 신체조성은 3등급 판정에만 쓴다. 응답에 수치로 나가지 않는다 (docs/02 §3).
        body_path = root / BODY_RANGES_FILE
        self.body_ranges: list[C.BodyRange] = (
            C.load_body_ranges_csv(body_path) if body_path.exists() else []
        )

        summary = pd.read_csv(root / "age_band_score_summary.csv", encoding="utf-8-sig")
        summary = summary[summary["status"] == "ok"]
        dist = pd.read_csv(root / "age_band_score_distribution.csv", encoding="utf-8-sig")
        quant = pd.read_csv(root / "age_band_value_quantiles.csv", encoding="utf-8-sig")
        qcols = [c for c in quant.columns if c.startswith("q")]

        bins = {
            k: g.sort_values("bin_lo")["ratio"].to_numpy(float)
            for k, g in dist.groupby(["age_group", "age_lo", "sex", "item_code"])
        }
        quants = {
            (r.age_group, r.age_lo, r.sex, r.item_code): np.sort(
                np.array([getattr(r, c) for c in qcols], dtype=float)
            )
            for r in quant.itertuples()
        }

        self._cells: dict[tuple[str, int, str, str], Cell] = {}
        self._bands: dict[str, list[tuple[int, int]]] = {}
        for r in summary.itertuples():
            code = f"{int(r.item_code):03d}"
            key = (r.age_group, int(r.age_lo), r.sex, int(r.item_code))
            xs, ys = [], []
            for y in (0, 40, 60, 80, 100):
                x = getattr(r, f"anchor_{y}")
                if pd.notna(x) and x != "":
                    xs.append(float(x))
                    ys.append(float(y))
            self._cells[(r.age_group, int(r.age_lo), r.sex, code)] = Cell(
                age_group=r.age_group,
                age_lo=int(r.age_lo),
                age_hi=int(r.age_hi),
                age_unit=r.age_unit,
                sex=r.sex,
                item_code=code,
                n=int(r.n),
                anchors=Anchors(x=tuple(xs), y=tuple(ys), method=r.method),
                quantiles=quants[key],
                score_bins=bins[key],
            )
            self._bands.setdefault(r.age_group, [])
            if (int(r.age_lo), int(r.age_hi)) not in self._bands[r.age_group]:
                self._bands[r.age_group].append((int(r.age_lo), int(r.age_hi)))

    def age_group_of(self, age: int, age_unit: str) -> str | None:
        """연령대는 산출물의 구간에서 되찾는다. 코드에 경계를 박아 두지 않는다."""
        for group, bands in self._bands.items():
            unit = "개월" if group == "유아기" else "세"
            if unit == age_unit and any(lo <= age <= hi for lo, hi in bands):
                return group
        return None

    def band_of(self, age_group: str, age: int) -> tuple[int, int] | None:
        for lo, hi in self._bands.get(age_group, []):
            if lo <= age <= hi:
                return (lo, hi)
        return None

    def cell(self, age_group: str, age: int, sex: str, item_code: str) -> Cell | None:
        band = self.band_of(age_group, age)
        return self._cells.get((age_group, band[0], sex, item_code)) if band else None

    def items_of(self, age_group: str, age: int, sex: str) -> list[str]:
        band = self.band_of(age_group, age)
        if not band:
            return []
        return sorted(c for (g, lo, s, c) in self._cells if (g, lo, s) == (age_group, band[0], sex))


@dataclass(frozen=True)
class FactorScore:
    factor: str
    item_code: str
    item_name: str
    unit: str
    value: float
    score: float
    percentile: int
    band: str
    n: int


@dataclass(frozen=True)
class Assessment:
    age_group: str | None
    sex: str
    band: tuple[int, int] | None = None
    input_level: str = "L1"
    low_sample: bool = False
    factors: list[FactorScore] = field(default_factory=list)
    focus_one: str | None = None
    not_scored: list[str] = field(default_factory=list)
    note: str | None = None
    grade: str | None = None
    grade_summary: str | None = None


def score_one(cell: Cell, value: float) -> FactorScore:
    """원값 하나를 점수·백분위·밴드로. 경험분포는 분위수 표에서 되살린다."""
    item = ITEMS[cell.item_code]
    s = float(
        score(
            np.array([value]),
            cell.anchors,
            cell.quantiles,
            lower_is_better=item.lower_is_better,
        )[0]
    )
    # 백분위는 원값 분위수에서 바로 구한다. 점수 히스토그램(10점 구간)을 거치면
    # 사람이 몰린 구간에서 최대 16%p 까지 뭉개진다.
    ordered = cell.quantiles if not item.lower_is_better else -cell.quantiles[::-1]
    probe = value if not item.lower_is_better else -value
    below = float(np.searchsorted(ordered, probe, side="right")) / ordered.size * 100
    percentile = int(round(below))
    band = (
        "strength"
        if percentile >= BAND_STRENGTH
        else ("growth" if percentile < BAND_GROWTH else "steady")
    )
    return FactorScore(
        factor=item.factor,
        item_code=cell.item_code,
        item_name=item.name,
        unit=item.unit,
        value=value,
        score=round(s, 1),
        percentile=percentile,
        band=band,
        n=cell.n,
    )


def assess(
    ref: Reference, *, age: int, age_unit: str, sex: str, measurements: dict[str, float]
) -> Assessment:
    age_group = ref.age_group_of(age, age_unit)
    if age_group is None:
        return Assessment(age_group=None, sex=sex, note="점수를 낼 수 있는 연령 구간이 아니다")

    factors: list[FactorScore] = []
    skipped: list[str] = []
    for code, value in measurements.items():
        cell = ref.cell(age_group, age, sex, code)
        if cell is None:
            skipped.append(code)
            continue
        factors.append(score_one(cell, value))
    factors.sort(key=lambda f: -f.score)
    # 등급은 점수와 따로 낸다 — 항목 AND 조건이라 요인 점수로 대신할 수 없다.
    verdict = (
        G.judge(
            ref.thresholds,
            age_group=age_group,
            age=age,
            sex=sex,
            measurements=measurements,
            body_ranges=ref.body_ranges,
        )
        if ref.thresholds
        else G.GradeResult(grade=None)
    )
    return Assessment(
        age_group=age_group,
        sex=sex,
        band=ref.band_of(age_group, age),
        input_level="L2" if factors else "L1",
        low_sample=any(f.n < 30 for f in factors),
        factors=factors,
        focus_one=factors[-1].factor if factors else None,
        not_scored=skipped,
        grade=verdict.grade,
        grade_summary=verdict.summary(),
    )


def _measure(text: str) -> tuple[str, float]:
    code, _, value = text.partition("=")
    return code.strip(), float(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="측정값을 점수와 분포상 위치로 바꾼다")
    ap.add_argument("--age", type=int, required=True, help="나이. 유아기는 개월 수")
    ap.add_argument("--age-unit", default="세", choices=["세", "개월"])
    ap.add_argument("--sex", required=True, choices=["M", "F"])
    ap.add_argument(
        "--measure",
        action="append",
        default=[],
        metavar="코드=값",
        help="예: --measure 028=52.3 (여러 번 쓴다)",
    )
    ap.add_argument("--release", default=str(DEFAULT_RELEASE), help="산출물 디렉터리")
    ap.add_argument("--json", action="store_true", help="사람이 읽는 표 대신 JSON")
    args = ap.parse_args(argv)

    ref = Reference(args.release)
    result = assess(
        ref,
        age=args.age,
        age_unit=args.age_unit,
        sex=args.sex,
        measurements=dict(_measure(m) for m in args.measure),
    )
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0

    if result.age_group is None or result.band is None:
        print(result.note or "점수를 낼 수 없다")
        return 1

    lo, hi = result.band
    sex_ko = "여" if args.sex == "F" else "남"
    print(f"{result.age_group} · {sex_ko} · {args.age}{args.age_unit}   기준 구간 {lo}~{hi}")
    print(f"\n  {'요인':<9}{'항목':<15}{'값':>8}{'점수':>8}{'또래 상위':>11}{'밴드':>11}")
    print("  " + "-" * 60)
    for f in result.factors:
        print(
            f"  {f.factor:<9}{f.item_name:<15}{f.value:>8}"
            f"{f.score:>8}{100 - f.percentile:>10}%{f.band:>11}"
        )
    print(f"\n  {result.grade_summary or '등급 판정 불가'}")
    if result.focus_one:
        print(f"  대상 요인 {result.focus_one}")
    if result.not_scored:
        print(f"  이 구간의 기준항목이 아니다: {', '.join(result.not_scored)}")
    if result.low_sample:
        print("  표본이 30 미만인 칸이 있다 (low_sample)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
