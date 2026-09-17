"""측정값 → 요인별 점수. 측정값이 없어도 200 이다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from family_fitness_ai.common import copy as words
from family_fitness_ai.common.items import (
    BODY_ITEMS,
    ITEMS,
    age_group_of,
    item_label,
    scored_items,
)
from family_fitness_ai.stats import tables


@dataclass(frozen=True)
class Profile:
    profile_ref: str
    age: int
    age_unit: str
    sex: str
    height_cm: float | None = None
    weight_kg: float | None = None
    measurements: dict[str, float] | None = None

    @property
    def age_group(self) -> str:
        return age_group_of(self.age, self.age_unit)

    @property
    def values(self) -> dict[str, float]:
        return self.measurements or {}

    @property
    def input_level(self) -> str:
        if self.values:
            return "L2"
        if self.height_cm is not None and self.weight_kg is not None:
            return "L1"
        return "L0"


def _with_body(profile: Profile) -> dict[str, float]:
    """등급 판정용 값. 키·몸무게만 있으면 BMI 는 우리가 낸다."""
    values = dict(profile.values)
    if "018" not in values and profile.height_cm and profile.weight_kg:
        metres = profile.height_cm / 100
        values["018"] = round(profile.weight_kg / (metres * metres), 2)
    if "042" not in values and "004" in values and profile.height_cm:
        values["042"] = round(values["004"] / profile.height_cm, 3)
    return values


def factor_rows(profile: Profile) -> tuple[list[dict[str, Any]], bool]:
    """요인별 한 줄씩. 두 번째 값은 표본이 모자란 칸이 있었는지다."""
    age_group = profile.age_group
    rows: list[dict[str, Any]] = []
    low_sample = False

    for code in scored_items(age_group, profile.values):
        item = ITEMS[code]
        value = float(profile.values[code])
        sample = tables.peer(age_group, profile.sex, profile.age, code)

        if sample is None or not sample.enough:
            low_sample = True
            score = percentile = band = None
            n = sample.n if sample else 0
        else:
            percentile = tables.percentile_of(sample, value, item.lower_is_better)
            score = tables.score_of(sample, value, item.lower_is_better)
            band = words.band_of(percentile)
            n = sample.n

        rows.append(
            {
                "factor": item.factor,
                "item_code": code,
                "item_name": item.name,
                "item_label": item_label(code, age_group),
                "unit": item.unit,
                "value": value,
                "score": score,
                "percentile": percentile,
                "band": band,
                "n": n,
            }
        )
    return rows, low_sample


def _pick(rows: list[dict[str, Any]], *, lowest: bool) -> dict[str, Any] | None:
    graded = [row for row in rows if row["percentile"] is not None]
    if not graded:
        return None
    return (min if lowest else max)(graded, key=lambda row: int(row["percentile"]))


def assessment(profile: Profile) -> dict[str, Any]:
    age_group = profile.age_group
    rows, low_sample = factor_rows(profile)

    focus = _pick(rows, lowest=True)
    best = _pick(rows, lowest=False)

    # 측정값이 없으면 요인을 지목하지 않는다. 근거 없이 고른 요인에는 인용할
    # 것이 없다.
    child_scope: dict[str, Any] = {"focus_one": None}
    parent_copy: dict[str, str] = {}
    if focus is not None:
        factor = str(focus["factor"])
        child_scope = {"focus_one": {"factor": factor, "copy": words.FOCUS_COPY[factor]}}
        parent_copy["focus"] = words.factor_copy(factor, str(focus["band"]))
    if best is not None:
        parent_copy["strength"] = words.factor_copy(str(best["factor"]), str(best["band"]))

    values = _with_body(profile)
    grade = tables.certify(age_group, profile.sex, profile.age, values) if rows else None

    return {
        "input_level": profile.input_level,
        "age_group": age_group,
        "child_scope": child_scope,
        "parent_scope": {
            "grade": grade,
            "peer_distribution": tables.grade_distribution(age_group, profile.sex, profile.age),
            "factors": rows,
            "copy": parent_copy,
        },
        "low_sample": low_sample,
        "disclaimer": words.DISCLAIMER,
    }


def trajectory(profile: Profile, item_code: str = "028", horizon_years: int = 10) -> dict[str, Any]:
    """또래 집단이 나이를 따라 보이는 분포. 개인의 미래가 아니다."""
    step = 12 if profile.age_unit == "개월" else 1
    ages = [profile.age + step * year for year in range(horizon_years + 1)]

    bands: list[dict[str, Any]] = []
    for age in ages:
        group = age_group_of(age, profile.age_unit)
        bands.extend(tables.trajectory_bands(group, profile.sex, item_code, [age]))

    item = ITEMS.get(item_code)
    name = item.name if item else BODY_ITEMS.get(item_code, (item_code, ""))[0]
    unit = item.unit if item else BODY_ITEMS.get(item_code, ("", ""))[1]

    return {
        "basis": "cross_sectional_group_distribution",
        "item_code": item_code,
        "item_name": name,
        "unit": unit,
        "bands": bands,
        "notice": words.TRAJECTORY_NOTICE,
        "low_sample": any(int(band["n"]) < tables.MIN_SAMPLE for band in bands) or not bands,
    }
