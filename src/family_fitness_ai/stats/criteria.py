"""등급 기준표(xlsx) → 문턱 표.

기준표는 연령대별 시트 하나씩이고, 등급·성별이 병합 셀로 묶여 있다. 여기서 하는 일은
그 격자를 한 행 = 한 문턱으로 펴는 것뿐이다. 방향(클수록 좋은가)은 기준표에 없으므로
`items.Item.lower_is_better` 가 정본이다 (docs/02 §5.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import openpyxl

# 연령대 값의 정본은 계약이다 (docs/03 §2.4). stats 가 다시 적지 않는다.
from ..common.types import SCORED_AGE_GROUPS
from . import items as I

_GRADE = re.compile(r"([123])\s*등급")
_SEX = {"남": "M", "여": "F"}
_AGE_MAX = 120  # '85이상' 같은 열린 구간의 상한. 분포 산출에서만 쓰는 값이다.


@dataclass(frozen=True)
class Threshold:
    age_group: str
    sex: str
    age_lo: int
    age_hi: int
    item_code: str
    grade: int
    value: float


def parse_age_band(raw: object) -> tuple[int, int] | None:
    """'48~53' · 11 · '19~24' · '85이상' → (lo, hi). 유아기는 개월이다."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        n = int(raw)
        return (n, n)
    text = str(raw).strip().replace(" ", "")
    if not text:
        return None
    if "이상" in text:
        lo = int(re.sub(r"\D", "", text))
        return (lo, _AGE_MAX)
    m = re.match(r"^(\d+)[~\-](\d+)$", text)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    if text.isdigit():
        n = int(text)
        return (n, n)
    return None


def _as_float(v: object) -> float | None:
    """숫자 칸만 받는다. '< 24.2' 같은 구간 형태는 점수화 대상이 아니다."""
    if isinstance(v, (int, float)):
        return float(v)
    return None


def load_xlsx(xlsx_path: str | Path) -> list[Threshold]:
    """기준표 원본(xlsx)에서 읽는다. 원본이 갱신될 때만 쓴다."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    out: list[Threshold] = []

    for sheet in wb.worksheets:
        age_group = I.SHEET_TO_AGE_GROUP.get(sheet.title)
        if age_group not in SCORED_AGE_GROUPS:
            continue
        columns = I.CRITERIA_COLUMNS[age_group]

        rows = list(sheet.iter_rows(values_only=True))
        header = [I.normalise_header(v) for v in rows[2]]
        col_of: dict[int, tuple[str, ...]] = {
            idx: codes for idx, text in enumerate(header) if (codes := columns.get(text))
        }
        missing = set(columns) - {header[i] for i in col_of}
        if missing:
            raise ValueError(f"{sheet.title}: 기준표 열을 찾지 못했다 — {sorted(missing)}")

        grade = sex = None
        for row in rows[3:]:
            if m := _GRADE.search(str(row[0] or "")):
                grade = int(m.group(1))
            if s := _SEX.get(str(row[1] or "").strip()):
                sex = s
            band = parse_age_band(row[2])
            if grade is None or sex is None or band is None:
                continue
            for idx, codes in col_of.items():
                value = _as_float(row[idx])
                if value is None:
                    continue
                for code in codes:
                    out.append(Threshold(age_group, sex, band[0], band[1], code, grade, value))
    return out


def to_frame(thresholds: list[Threshold]):
    """문턱을 평평한 표로. 기준표 원본(xlsx) 없이도 값을 확인할 수 있게 한다."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "age_group": t.age_group,
                "sex": t.sex,
                "age_lo": t.age_lo,
                "age_hi": t.age_hi,
                "age_unit": "개월" if t.age_group == "유아기" else "세",
                "item_code": t.item_code,
                "item_name": I.ITEMS[t.item_code].name,
                "unit": I.ITEMS[t.item_code].unit,
                "lower_is_better": int(I.ITEMS[t.item_code].lower_is_better),
                "grade": t.grade,
                "threshold": t.value,
            }
            for t in thresholds
        ]
    ).sort_values(["age_group", "sex", "age_lo", "item_code", "grade"])


def load_csv(csv_path: str | Path) -> list[Threshold]:
    """레포에 커밋된 문턱 표에서 읽는다. 원본 xlsx 가 없어도 산출이 돈다."""
    import csv as _csv

    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        return [
            Threshold(
                age_group=r["age_group"],
                sex=r["sex"],
                age_lo=int(r["age_lo"]),
                age_hi=int(r["age_hi"]),
                item_code=r["item_code"],
                grade=int(r["grade"]),
                value=float(r["threshold"]),
            )
            for r in _csv.DictReader(fh)
        ]


def load(path: str | Path) -> list[Threshold]:
    """확장자로 원본(xlsx)과 커밋본(csv)을 가른다."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix in (".xlsx", ".xlsm"):
        return load_xlsx(path)
    raise ValueError(f"기준표는 .xlsx 또는 .csv 여야 한다: {path}")
