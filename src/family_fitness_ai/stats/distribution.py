"""연령 구간별 점수 분포.

측정값을 요인 점수(0~100)로 바꾼 뒤, 기준표가 정의한 연령 구간마다 분포를 낸다.
구간을 기준표에 맞추는 이유는 각 구간이 자기 문턱을 갖기 때문이다 — 문턱이 다르면
같은 raw 값이 다른 점수가 되고, 그것이 정규화의 요점이다 (docs/02 §5.1).

실행 (저장소 루트에서):
    python -m family_fitness_ai.stats.distribution --data-dir <원자료 디렉터리>

기준표는 레포에 커밋된 `data/release/grade_thresholds.csv` 를 기본으로 쓴다. 원본이
갱신되면 `--criteria <등급평가항목및기준.xlsx>` 로 다시 만든다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest import measurements as M
from . import criteria as C
from .items import ITEMS
from .score import ANCHOR_SCORES, build_anchors, score

# 기준표는 레포에 커밋되어 있다. 원자료 zip 에는 측정 기록만 담긴다 (docs/02 §3.1).
DEFAULT_CRITERIA = Path("data/release/grade_thresholds.csv")

BIN_EDGES = np.arange(0, 101, 10)
QUANTILES = (10, 25, 50, 75, 90)


def _bands(thresholds: list[C.Threshold], age_group: str) -> list[tuple[int, int]]:
    return sorted({(t.age_lo, t.age_hi) for t in thresholds if t.age_group == age_group})


def build(df: pd.DataFrame, thresholds: list[C.Threshold]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(요약, 분포) 두 표를 낸다."""
    index: dict[tuple[str, str, int, int, str], dict[int, float]] = {}
    for t in thresholds:
        index.setdefault((t.age_group, t.sex, t.age_lo, t.age_hi, t.item_code), {})[t.grade] = (
            t.value
        )

    summary_rows: list[dict] = []
    dist_rows: list[dict] = []

    for age_group, group_df in df.groupby(M.AGE_GROUP_COL, sort=False):
        for lo, hi in _bands(thresholds, str(age_group)):
            band_df = group_df[(group_df[M.AGE_COL] >= lo) & (group_df[M.AGE_COL] <= hi)]
            if band_df.empty:
                continue
            for sex in ("M", "F"):
                cell = band_df[band_df[M.SEX_COL] == sex]
                if cell.empty:
                    continue
                for code, item in ITEMS.items():
                    col = M._item_column(code)
                    if col not in cell.columns:
                        continue
                    values = cell[col].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    th = index.get((str(age_group), sex, lo, hi, code))
                    if th is None:
                        continue  # 기준표에 없는 항목은 그 칸의 평가 대상이 아니다
                    anchors, reason = build_anchors(
                        values, th, lower_is_better=item.lower_is_better
                    )
                    scores = (
                        score(values, anchors, values, lower_is_better=item.lower_is_better)
                        if anchors is not None
                        else np.array([])
                    )
                    scores = scores[np.isfinite(scores)]

                    base = {
                        "age_group": age_group,
                        "age_lo": lo,
                        "age_hi": hi,
                        "age_unit": "개월" if age_group == "유아기" else "세",
                        "sex": sex,
                        "factor": item.factor,
                        "item_code": code,
                        "item_name": item.name,
                    }
                    ok = anchors is not None and scores.size > 0
                    summary_rows.append(
                        {
                            **base,
                            "unit": item.unit,
                            "lower_is_better": int(item.lower_is_better),
                            "status": reason if not ok else "ok",
                            "n_measured": int(values.size),
                            "n": int(scores.size),
                            "method": anchors.method if anchors else "",
                            "grade3_from_p25": int(anchors.grade3_from_p25) if anchors else "",
                            **{
                                f"anchor_{int(y)}": (round(float(x), 4) if anchors else "")
                                for x, y in zip(
                                    anchors.x if anchors else (0.0,) * 5,
                                    ANCHOR_SCORES,
                                    strict=True,
                                )
                            },
                            "value_min": round(float(values.min()), 4) if values.size else "",
                            "value_max": round(float(values.max()), 4) if values.size else "",
                            **{
                                f"score_p{q}": (
                                    round(float(np.percentile(scores, q)), 2) if ok else ""
                                )
                                for q in QUANTILES
                            },
                            "score_mean": round(float(scores.mean()), 2) if ok else "",
                        }
                    )
                    if not ok:
                        continue
                    counts, _ = np.histogram(scores, bins=BIN_EDGES)
                    counts[-1] += int((scores >= 100).sum())
                    total = int(counts.sum())
                    for i, count in enumerate(counts):
                        dist_rows.append(
                            {
                                **base,
                                "bin_lo": int(BIN_EDGES[i]),
                                "bin_hi": int(BIN_EDGES[i + 1]),
                                "count": int(count),
                                "ratio": round(count / total, 4) if total else 0.0,
                            }
                        )

    return pd.DataFrame(summary_rows), pd.DataFrame(dist_rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="연령 구간별 점수 분포를 CSV로 낸다")
    ap.add_argument("--data-dir", required=True, help="원자료 월별 CSV 디렉터리")
    ap.add_argument(
        "--criteria",
        default=str(DEFAULT_CRITERIA),
        help=f"등급 문턱. 기본값 {DEFAULT_CRITERIA} (원본 xlsx 도 받는다)",
    )
    ap.add_argument("--out", default="data/release", help="출력 디렉터리")
    args = ap.parse_args(argv)

    if not Path(args.criteria).exists():
        ap.error(
            f"기준표를 찾지 못했다: {args.criteria}\n"
            "저장소 루트에서 실행하거나 --criteria 로 경로를 준다."
        )

    df = M.load_dir(args.data_dir)
    thresholds = C.load(args.criteria)
    summary, dist = build(df, thresholds)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # utf-8-sig — 검수하는 사람이 엑셀로 연다 (docs/02 §4)
    summary.to_csv(out / "age_band_score_summary.csv", index=False, encoding="utf-8-sig")
    dist.to_csv(out / "age_band_score_distribution.csv", index=False, encoding="utf-8-sig")
    C.to_frame(thresholds).to_csv(out / "grade_thresholds.csv", index=False, encoding="utf-8-sig")

    skipped = int((summary["status"] != "ok").sum())
    if skipped:
        print(f"⚠ 눈금이 서지 않아 제외한 칸 {skipped}개 — 사유는 요약 CSV 의 status 열")
    print(f"원자료 {len(df):,}행 · 문턱 {len(thresholds):,}건")
    print(f"요약 {len(summary):,}행 → {out / 'age_band_score_summary.csv'}")
    print(f"분포 {len(dist):,}행 → {out / 'age_band_score_distribution.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
