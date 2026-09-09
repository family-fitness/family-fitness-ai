"""공공데이터 원자료(월별 CSV) 적재.

원자료는 저장소에 넣지 않는다. 목록은 `data/manifest.csv` 다 (docs/02 §3.1).
가공하지 않은 공단 배포본을 그대로 읽고, 필요한 열만 남긴다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.types import SCORED_AGE_GROUPS
from ..stats.items import ITEMS

AGE_GROUP_COL = "AGRDE_FLAG_NM"
AGE_COL = "MESURE_AGE_CO"  # 유아기만 개월, 나머지는 만 나이 (docs/02 §2.4)
SEX_COL = "SEXDSTN_FLAG_CD"
GRADE_COL = "CRTFC_FLAG_NM"
DATE_COL = "MESURE_DE"

# 점수화하지 않지만 3등급 판정에 쓴다 (docs/02 §5.2 vs §5.4). 응답에 수치로
# 나가지 않는다 (docs/02 §3).
BODY_COMPOSITION_CODES = ("003", "018", "042")


def _item_column(code: str) -> str:
    return f"MESURE_IEM_{code}_VALUE"


def _no_csv_message(root: Path) -> str:
    """CSV를 못 찾았을 때, 무엇을 봤는지 알려준다. 경로 오타와 zip 구조를 가른다."""
    if not root.exists():
        return f"디렉터리가 없다: {root}"
    entries = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
    nested = sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.csv")))
    lines = [f"CSV가 없다: {root}"]
    lines.append(f"  안에 있는 것: {', '.join(entries[:8]) or '(비어 있다)'}")
    if nested:
        lines.append(f"  CSV는 하위 디렉터리에 있다. --data-dir {root / nested[0]}")
    return "\n".join(lines)


def load_dir(data_dir: str | Path) -> pd.DataFrame:
    """디렉터리의 월별 CSV를 전부 읽어 한 표로 만든다.

    기간별로 나뉜 배포본을 한 디렉터리에 모아 두면 된다. 파일명 순으로 읽고 이어
    붙이므로 개수와 기간은 자유다. 하위 디렉터리는 뒤지지 않는다 — 어디서 읽었는지가
    분명해야 산출물의 기준 기간을 말할 수 있다.
    """
    root = Path(data_dir).expanduser()
    paths = sorted(root.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(_no_csv_message(root))

    keep = [AGE_GROUP_COL, AGE_COL, SEX_COL, GRADE_COL, DATE_COL]
    item_cols = [_item_column(c) for c in (*ITEMS, *BODY_COMPOSITION_CODES)]

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
