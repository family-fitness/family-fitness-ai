"""공공데이터 원자료(월별 CSV) 적재.

원자료는 저장소에 넣지 않는다. 목록은 `data/manifest.csv` 다 (docs/02 §3.1).
가공하지 않은 공단 배포본을 그대로 읽고, 필요한 열만 남긴다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..stats.items import ITEMS, SCORED_AGE_GROUPS

AGE_GROUP_COL = "AGRDE_FLAG_NM"
AGE_COL = "MESURE_AGE_CO"  # 유아기만 개월, 나머지는 만 나이 (docs/02 §2.4)
SEX_COL = "SEXDSTN_FLAG_CD"
GRADE_COL = "CRTFC_FLAG_NM"
DATE_COL = "MESURE_DE"


def _item_column(code: str) -> str:
    return f"MESURE_IEM_{code}_VALUE"


def load_dir(data_dir: str | Path) -> pd.DataFrame:
    """디렉터리의 월별 CSV를 전부 읽어 한 표로 만든다."""
    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"CSV가 없다: {data_dir}")

    keep = [AGE_GROUP_COL, AGE_COL, SEX_COL, GRADE_COL, DATE_COL]
    item_cols = [_item_column(c) for c in ITEMS]

    frames = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        cols = keep + [c for c in item_cols if c in df.columns]
        frames.append(df[cols])

    out = pd.concat(frames, ignore_index=True)
    out = out[out[AGE_GROUP_COL].isin(SCORED_AGE_GROUPS)]
    out = out[out[SEX_COL].isin(["M", "F"])]
    out[AGE_COL] = pd.to_numeric(out[AGE_COL], errors="coerce")
    out = out[out[AGE_COL].notna()]
    for col in item_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)
