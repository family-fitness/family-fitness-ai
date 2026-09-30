"""data/release 표를 읽어 들고 있는다. 원자료는 서비스에서 읽지 않는다."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from family_fitness_ai.common.settings import settings

#: 이보다 적은 표본은 점수를 내지 않는다. 낼 수 있는 척하지 않는다.
MIN_SAMPLE = 30


@dataclass(frozen=True)
class Peer:
    """한 (연령대·성별·나이·항목)의 또래 분포."""

    n: int
    mean: float
    sd: float
    quantiles: np.ndarray  # 101칸. 0퍼센타일부터 100퍼센타일까지.

    @property
    def enough(self) -> bool:
        return self.n >= MIN_SAMPLE


@dataclass(frozen=True)
class Threshold:
    age_lo: int
    age_hi: int
    item_code: str
    op: str  # >= · <= · < · between
    value: float
    value2: float

    def passes(self, value: float) -> bool:
        if self.op == ">=":
            return value >= self.value
        if self.op == "<=":
            return value <= self.value
        if self.op == "<":
            return value < self.value
        return self.value <= value <= self.value2


@lru_cache
def _peers() -> dict[tuple[str, str, int, str], Peer]:
    out: dict[tuple[str, str, int, str], Peer] = {}
    path = settings().release_dir / "value_quantiles.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["age_group"], row["sex"], int(row["age"]), row["item_code"])
            out[key] = Peer(
                n=int(row["n"]),
                mean=float(row["mean"]),
                sd=float(row["sd"]),
                quantiles=np.fromstring(row["quantiles"], sep=";"),
            )
    return out


@lru_cache
def _thresholds() -> dict[tuple[str, str, str], tuple[Threshold, ...]]:
    out: dict[tuple[str, str, str], list[Threshold]] = {}
    path = settings().release_dir / "grade_thresholds.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["age_group"], row["sex"], row["grade"])
            out.setdefault(key, []).append(
                Threshold(
                    age_lo=int(row["age_lo"]),
                    age_hi=int(row["age_hi"]),
                    item_code=row["item_code"],
                    op=row["op"],
                    value=float(row["value"]),
                    value2=float(row["value2"]),
                )
            )
    return {key: tuple(value) for key, value in out.items()}


@lru_cache
def _distribution() -> dict[tuple[str, str, int], tuple[tuple[str, float], ...]]:
    out: dict[tuple[str, str, int], list[tuple[str, float]]] = {}
    path = settings().release_dir / "grade_distribution.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["age_group"], row["sex"], int(row["age"]))
            out.setdefault(key, []).append((row["grade"], float(row["ratio"])))
    return {key: tuple(value) for key, value in out.items()}


def peer(age_group: str, sex: str, age: int, item_code: str) -> Peer | None:
    return _peers().get((age_group, sex, age, item_code))


def percentile_of(sample: Peer, value: float, lower_is_better: bool) -> int:
    """또래 중 내 자리. 0~100.

    같은 값이 잔뜩 몰린 항목(벽패스처럼 0이 많은 것)에서 순위가 확 튀지 않도록
    아래쪽 자리와 위쪽 자리의 가운데를 쓴다.
    """
    quantiles = sample.quantiles
    low = int(np.searchsorted(quantiles, value, side="left"))
    high = int(np.searchsorted(quantiles, value, side="right"))
    rank = (low + high) / 2
    if lower_is_better:
        rank = 100 - rank
    return int(round(min(100.0, max(0.0, rank))))


def score_of(sample: Peer, value: float, lower_is_better: bool) -> float:
    """0~100 점수.

    또래 평균·표준편차로 잰 자리를 정규분포로 편 값이다. 백분위가 실제 순위라면
    점수는 그 순위를 매끄럽게 편 눈금이라, 값이 조금 오르면 점수도 조금 오른다.
    """
    if sample.sd <= 0:
        return 50.0
    z = (value - sample.mean) / sample.sd
    if lower_is_better:
        z = -z
    score = 100 * 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return round(min(100.0, max(0.0, score)), 1)


def grade_thresholds(age_group: str, sex: str, grade: str, age: int) -> list[Threshold]:
    rows = _thresholds().get((age_group, sex, grade), ())
    return [row for row in rows if row.age_lo <= age <= row.age_hi]


def grade_distribution(age_group: str, sex: str, age: int) -> list[dict[str, object]]:
    rows = _distribution().get((age_group, sex, age), ())
    return [{"grade": grade, "ratio": ratio} for grade, ratio in rows]


def certify(age_group: str, sex: str, age: int, values: dict[str, float]) -> str | None:
    """국민체력100 인증 등급.

    한 등급으로 인증하려면 그 등급이 보는 항목을 **전부 재고 전부 충족**해야
    한다. 하나라도 안 쟀으면 그 등급으로는 판정하지 않는다 — 집에서 두어 개만
    잰 사람을 맨 아래로 내리지 않으려는 것이다. 전부 쟀는데 못 미치면 `참가`,
    잴 항목이 부족하거나 기준표가 없으면 `None` 이다.
    """
    judged = False
    for grade in ("1등급", "2등급", "3등급"):
        rows = grade_thresholds(age_group, sex, grade, age)
        if not rows:
            continue
        # 심폐지구력처럼 두 시험 중 하나만 재는 항목이 있다. 같은 요인끼리 묶어
        # 하나라도 통과하면 그 자리는 통과로 본다.
        by_item = {row.item_code: row for row in rows}
        measured = {code: row for code, row in by_item.items() if code in values}
        alternatives = {"035", "037"}
        required = set(by_item) - alternatives
        if alternatives & set(by_item) and not (alternatives & set(measured)):
            continue
        if not required <= set(measured):
            continue
        judged = True
        ok = all(row.passes(values[code]) for code, row in measured.items())
        if ok:
            return grade
    return "참가" if judged else None


def trajectory_bands(
    age_group: str, sex: str, item_code: str, ages: list[int]
) -> list[dict[str, object]]:
    """나이를 따라가며 또래 분포가 어디에 놓이는지. 개인의 미래가 아니다."""
    bands: list[dict[str, object]] = []
    for age in ages:
        sample = peer(age_group, sex, age, item_code)
        if sample is None:
            continue
        bands.append(
            {
                "age": age,
                "p10": round(float(sample.quantiles[10]), 1),
                "p50": round(float(sample.quantiles[50]), 1),
                "p90": round(float(sample.quantiles[90]), 1),
                "n": sample.n,
            }
        )
    return bands
