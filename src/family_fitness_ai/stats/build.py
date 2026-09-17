"""원자료 → data/release 표 세 장.

    python -m family_fitness_ai.stats.build --data-dir data/raw

- value_quantiles.csv    또래 분포. 백분위·점수·궤적이 전부 여기서 나온다.
- grade_thresholds.csv   국민체력100 인증 기준. 등급 판정이 여기서 나온다.
- grade_distribution.csv 또래가 실제로 받은 등급의 비율.

서비스는 원자료를 읽지 않는다. 표만 읽는다.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from family_fitness_ai.common.items import BODY_ITEMS, ITEMS
from family_fitness_ai.common.settings import ROOT

DATASET = "dataset/kspo-data-set-measure-prescript"
CRITERIA_XLSX = "등급평가항목및기준.xlsx"

#: 기준표 시트 이름 → 연령대. 노년기는 시험 자체가 다른 표를 쓴다 — 이번 범위 밖.
SHEETS = {"유아기": "유아기", "유소년": "유소년", "청소년": "청소년", "성인": "성인"}

#: 기준표 항목명(공백 지운 것) → 항목 코드. 같은 이름이 연령대마다 다른 시험인
#: 경우가 있어 연령대별로 가른다.
_HEADER_CODES: dict[str, dict[str, tuple[str, ...]]] = {
    "유아기": {
        "10m왕복오래달리기(회)": ("020",),
        "상대악력(%)": ("028",),
        "윗몸말아올리기(회)": ("009",),
        "앉아윗몸앞으로굽히기(cm)": ("012",),
        "5mX4왕복달리기(초)": ("050",),
        "제자리멀리뛰기(cm)": ("022",),
        "3X3버튼누르기(초)": ("051",),
        "BMI(㎏/㎡)": ("018",),
    },
    "유소년": {
        "15m왕복오래달리기(회)": ("020",),
        "상대악력(%)": ("028",),
        "윗몸말아올리기(회)": ("009",),
        "앉아윗몸앞으로굽히기(cm)": ("012",),
        "반복옆뛰기(회)": ("043",),
        "제자리멀리뛰기(cm)": ("022",),
        "눈-손협응력검사(회)": ("044",),
        "BMI(㎏/㎡)": ("018",),
        "허리둘레-신장비(WHtR)": ("042",),
    },
    "청소년": {
        "20m왕복오래달리기(회)": ("020",),
        "트레드밀/스텝검사(ml/kg/min)": ("035", "037"),
        "상대악력(%)": ("028",),
        "윗몸말아올리기(회)": ("009",),
        "반복점프(회)": ("010",),
        "앉아윗몸앞으로굽히기(cm)": ("012",),
        "일리노이(초)": ("013",),
        "체공시간(초)": ("014",),
        "눈-손협응력검사(초)": ("017",),
        "BMI(㎏/㎡)": ("018",),
        "체지방률(%)": ("003",),
    },
    "성인": {
        "20m왕복오래달리기(회)": ("020",),
        "트레드밀/스텝검사(ml/kg/min)": ("035", "037"),
        "상대악력(%)": ("028",),
        "교차윗몸일으키기(회)": ("019",),
        "앉아윗몸앞으로굽히기(cm)": ("012",),
        "10미터왕복달리기(초)": ("021",),
        "반응시간(초)": ("040",),
        "제자리멀리뛰기(cm)": ("022",),
        "체공시간(초)": ("041",),
        "BMI(㎏/㎡)": ("018",),
        "체지방률(%)": ("003",),
    },
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def _age_span(text: str) -> tuple[int, int]:
    """'48~53' · '11' · '85이상' → (lo, hi)."""
    t = _norm(text)
    if "이상" in t:
        return int(re.sub(r"\D", "", t)), 999
    if "~" in t:
        lo, hi = t.split("~", 1)
        return int(lo), int(hi)
    return int(t), int(t)


def _threshold(text: str) -> tuple[str, float, float] | None:
    """기준 칸 한 개 → (연산, 값, 값2).

    '46.5' → 항목 방향에 따라 >= 또는 <=
    '< 24.2' · '<17.5' → 아래여야 통과
    '18.5이상 25미만' · '7%초과 27%미만' → 두 값 사이
    """
    t = _norm(text)
    if not t:
        return None
    nums = [float(n) for n in re.findall(r"\d*\.?\d+", t)]
    if not nums:
        return None
    if len(nums) >= 2 and ("미만" in t or "이하" in t):
        return "between", nums[0], nums[1]
    if t.startswith("<") or "미만" in t:
        return "<", nums[0], 0.0
    return "plain", nums[0], 0.0


def build_thresholds(data_dir: Path, out_dir: Path) -> int:
    wb = openpyxl.load_workbook(data_dir / DATASET / CRITERIA_XLSX)
    rows: list[dict[str, object]] = []

    for sheet, age_group in SHEETS.items():
        ws = wb[sheet]
        table = [list(r) for r in ws.iter_rows(values_only=True)]
        header = [_norm(c) if c is not None else "" for c in table[2]]
        codes = _HEADER_CODES[age_group]
        age_unit = "개월" if age_group == "유아기" else "세"

        grade = sex = ""
        # 3등급의 신체조성 칸은 성인처럼 첫 줄에만 적힌 경우가 있다. 같은
        # (등급·성별) 안에서 마지막 값을 이어 쓴다.
        carried: dict[str, str] = {}
        for raw in table[3:]:
            cells = [("" if c is None else str(c)) for c in raw]
            if not any(_norm(c) for c in cells):
                continue
            grade = re.sub(r"\(.*?\)", "", _norm(cells[0])) or grade
            new_sex = _norm(cells[1])
            if new_sex:
                sex = new_sex
                carried = {}
            if not cells[2].strip():
                continue
            age_lo, age_hi = _age_span(cells[2])

            for col, name in enumerate(header):
                if name not in codes:
                    continue
                cell = cells[col] if col < len(cells) else ""
                is_body = codes[name][0] in BODY_ITEMS
                if is_body and not _norm(cell):
                    cell = carried.get(name, "")
                elif is_body:
                    carried[name] = cell
                parsed = _threshold(cell)
                if parsed is None:
                    continue
                op, value, value2 = parsed
                for code in codes[name]:
                    if op == "plain":
                        item = ITEMS.get(code)
                        op_out = "<=" if item and item.lower_is_better else ">="
                    else:
                        op_out = op
                    rows.append(
                        {
                            "age_group": age_group,
                            "sex": "M" if sex == "남" else "F",
                            "age_unit": age_unit,
                            "age_lo": age_lo,
                            "age_hi": age_hi,
                            "grade": grade,
                            "item_code": code,
                            "op": op_out,
                            "value": value,
                            "value2": value2,
                        }
                    )

    path = out_dir / "grade_thresholds.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _read_raw(data_dir: Path) -> pd.DataFrame:
    files = sorted((data_dir / DATASET).glob("*.csv"))
    if not files:
        raise SystemExit(f"원자료가 없다: {data_dir / DATASET}")
    keep = ["AGRDE_FLAG_NM", "MESURE_AGE_CO", "SEXDSTN_FLAG_CD", "CRTFC_FLAG_NM"]
    value_cols = [f"MESURE_IEM_{code}_VALUE" for code in ITEMS]
    frames = []
    for path in files:
        frame = pd.read_csv(path, usecols=keep + value_cols, encoding="utf-8-sig", low_memory=False)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(
        columns={
            "AGRDE_FLAG_NM": "age_group",
            "MESURE_AGE_CO": "age",
            "SEXDSTN_FLAG_CD": "sex",
            "CRTFC_FLAG_NM": "grade",
        }
    )
    raw["age"] = pd.to_numeric(raw["age"], errors="coerce")
    return raw.dropna(subset=["age", "sex", "age_group"])


def build_quantiles(raw: pd.DataFrame, out_dir: Path) -> int:
    grid = np.arange(101)
    rows: list[dict[str, object]] = []
    for code in ITEMS:
        col = f"MESURE_IEM_{code}_VALUE"
        values = pd.to_numeric(raw[col], errors="coerce")
        frame = raw.loc[values.notna(), ["age_group", "sex", "age"]].copy()
        if frame.empty:
            continue
        frame["value"] = values[values.notna()].to_numpy()
        for (age_group, sex, age), group in frame.groupby(["age_group", "sex", "age"]):
            sample = group["value"].to_numpy()
            quantiles = np.percentile(sample, grid)
            rows.append(
                {
                    "age_group": age_group,
                    "sex": sex,
                    "age_unit": "개월" if age_group == "유아기" else "세",
                    "age": int(age),
                    "item_code": code,
                    "n": len(sample),
                    "mean": round(float(sample.mean()), 4),
                    "sd": round(float(sample.std(ddof=1)) if len(sample) > 1 else 0.0, 4),
                    "quantiles": ";".join(f"{q:.4g}" for q in quantiles),
                }
            )

    path = out_dir / "value_quantiles.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        ordered = sorted(rows, key=lambda r: (r["age_group"], r["sex"], r["item_code"], r["age"]))
        writer.writerows(ordered)
    return len(rows)


def build_distribution(raw: pd.DataFrame, out_dir: Path) -> int:
    frame = raw.dropna(subset=["grade"])
    rows: list[dict[str, object]] = []
    for (age_group, sex, age), group in frame.groupby(["age_group", "sex", "age"]):
        total = len(group)
        counts = group["grade"].value_counts()
        for grade in ("1등급", "2등급", "3등급", "참가"):
            count = int(counts.get(grade, 0))
            rows.append(
                {
                    "age_group": age_group,
                    "sex": sex,
                    "age": int(age),
                    "grade": grade,
                    "n": count,
                    "ratio": round(count / total, 4),
                }
            )

    path = out_dir / "grade_distribution.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "release"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"기준표 {build_thresholds(data_dir, out_dir)}행")
    raw = _read_raw(data_dir)
    print(f"원자료 {len(raw):,}행")
    print(f"또래 분포 {build_quantiles(raw, out_dir)}행")
    print(f"등급 분포 {build_distribution(raw, out_dir)}행")


if __name__ == "__main__":
    main()
