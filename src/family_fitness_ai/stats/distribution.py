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
from . import grade as G
from .items import ITEMS
from .score import Anchors, build_anchors, score

# 기준표는 레포에 커밋되어 있다. 원자료 zip 에는 측정 기록만 담긴다 (docs/02 §3.1).
DEFAULT_CRITERIA = Path("data/release/grade_thresholds.csv")
# 신체조성은 문턱이 아니라 구간이라 표를 따로 둔다 (docs/dev/AI-2 §6).
BODY_RANGES_FILE = "body_composition_ranges.csv"


def _body_ranges(args) -> list:  # noqa: ANN001 — argparse Namespace
    """원본 xlsx 가 있으면 거기서, 없으면 커밋된 표에서 읽는다."""
    if Path(args.criteria).suffix.lower() in (".xlsx", ".xlsm"):
        return C.load_body_ranges_xlsx(args.criteria)
    committed = Path(args.out) / BODY_RANGES_FILE
    return C.load_body_ranges_csv(committed) if committed.exists() else []


BIN_EDGES = np.arange(0, 101, 10)
QUANTILES = (10, 25, 50, 75, 90)

# 칸별 원값 분위수. 서비스가 원자료 없이 경험분포를 되살려 채점하는 데 쓴다.
# 99점을 다 싣는 이유는 꼬리를 자르면 최악 오차가 20점까지 벌어지기 때문이다.
VALUE_QUANTILES = np.arange(1, 100)


def _anchor_columns(anchors: Anchors | None) -> dict[str, object]:
    """앵커를 고정된 열로 편다. 3등급 문턱이 없는 항목은 anchor_40 이 빈칸이다."""
    got = dict(zip(anchors.y, anchors.x, strict=True)) if anchors is not None else {}
    return {
        f"anchor_{k}": (round(float(got[k]), 4) if k in got else "") for k in (0, 40, 60, 80, 100)
    }


def _bands(thresholds: list[C.Threshold], age_group: str) -> list[tuple[int, int]]:
    return sorted({(t.age_lo, t.age_hi) for t in thresholds if t.age_group == age_group})


def build(
    df: pd.DataFrame, thresholds: list[C.Threshold]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(요약, 분포, 원값 분위수) 세 표를 낸다."""
    index: dict[tuple[str, str, int, int, str], dict[int, float]] = {}
    for t in thresholds:
        index.setdefault((t.age_group, t.sex, t.age_lo, t.age_hi, t.item_code), {})[t.grade] = (
            t.value
        )

    summary_rows: list[dict] = []
    dist_rows: list[dict] = []
    quantile_rows: list[dict] = []

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
                            "has_grade3": int(anchors.has_grade3) if anchors else "",
                            **_anchor_columns(anchors),
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
                    quantile_rows.append(
                        {
                            **{
                                k: base[k]
                                for k in ("age_group", "age_lo", "age_hi", "age_unit", "sex")
                            },
                            "item_code": code,
                            "n": int(values.size),
                            **{
                                f"q{q:02d}": round(float(x), 4)
                                for q, x in zip(
                                    VALUE_QUANTILES,
                                    np.percentile(values, VALUE_QUANTILES),
                                    strict=True,
                                )
                            },
                        }
                    )
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

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(dist_rows),
        pd.DataFrame(quantile_rows),
    )


# 원자료에 실제로 있는 등급 전부. 4·5·6등급은 2025-06 개편으로 생겼고 청소년·성인·
# 어르신에만 나타난다 — 빠뜨리면 그 연령대 분포에서 18만 행이 통째로 사라진다.
GRADE_ORDER = ("1등급", "2등급", "3등급", "4등급", "5등급", "6등급", G.PARTICIPATED)

# 2025-06 등급체계 개편. 그 전에는 4·5·6등급이 없어 지금의 4~6등급에 해당하는 사람이
# 전부 `참가` 로 기록됐다. **두 제도를 섞어 분포를 내면 안 된다** — 섞으면 `참가` 가
# 부풀고, 자기 등급을 그 분포 위에 올려 읽는 것이 틀린다 (docs/dev/AI-2 §5.3).
REFORM_YM = "202506"


def current_standard(df: pd.DataFrame) -> pd.DataFrame:
    """현행 등급체계로 기록된 행만. 기준 기간을 화면·리포트에 명시한다 (docs/02 §1.2)."""
    return df[df[M.DATE_COL].astype(str).str[:6] >= REFORM_YM]


def build_grade_distribution(df: pd.DataFrame, thresholds: list[C.Threshold]) -> pd.DataFrame:
    """또래 등급 분포. **판정을 다시 돌리지 않고 원자료의 등급 컬럼을 센다.**

    공단이 기록한 등급이 원자료에 이미 있다. 우리 판정으로 분포를 만들면 판정
    로직의 오차가 분포에 실리고, 사용자는 "내 등급"과 "또래 분포"를 같은 잣대로
    읽지 못한다 (docs/dev/AI-2 §2).
    """
    rows: list[dict] = []
    df = current_standard(df)
    for age_group, group_df in df.groupby(M.AGE_GROUP_COL, sort=False):
        for lo, hi in _bands(thresholds, str(age_group)):
            band_df = group_df[(group_df[M.AGE_COL] >= lo) & (group_df[M.AGE_COL] <= hi)]
            for sex in ("M", "F"):
                cell = band_df[band_df[M.SEX_COL] == sex]
                # 등급이 비어 있는 행은 분모에서도 뺀다. 미판정을 참가로 세면
                # 참가 비율이 부풀어 오른다 (docs/dev/AI-2 §4).
                graded = cell[cell[M.GRADE_COL].isin(GRADE_ORDER)]
                total = int(len(graded))
                if not total:
                    continue
                counts = graded[M.GRADE_COL].value_counts()
                for name in GRADE_ORDER:
                    count = int(counts.get(name, 0))
                    rows.append(
                        {
                            "age_group": age_group,
                            "age_lo": lo,
                            "age_hi": hi,
                            "age_unit": "개월" if age_group == "유아기" else "세",
                            "sex": sex,
                            "grade": name,
                            "count": count,
                            "ratio": round(count / total, 4),
                            "n_cell": total,
                        }
                    )
    return pd.DataFrame(rows)


def concordance(
    df: pd.DataFrame,
    thresholds: list[C.Threshold],
    body_ranges: list[C.BodyRange] | None = None,
    *,
    sample: int = 20000,
) -> dict:
    """우리 판정과 공단 기록이 얼마나 같은가. **판정 로직의 검사다.**

    전수를 돌리면 오래 걸리므로 표본을 본다. 파일로 만들지 않는다 — 매번 달라지는
    진단값이고 커밋할 산출물이 아니다 (docs/dev/AI-2 §4).
    """
    graded = current_standard(df)
    graded = graded[graded[M.GRADE_COL].isin(GRADE_ORDER)]
    if graded.empty:
        return {"n": 0}
    if len(graded) > sample:
        graded = graded.sample(sample, random_state=0)

    item_columns = {code: M._item_column(code) for code in (*ITEMS, *M.BODY_COMPOSITION_CODES)}
    agree = disagree_generous = disagree_strict = undecidable = 0
    for row in graded.itertuples(index=False):
        values = {
            code: float(getattr(row, col))
            for code, col in item_columns.items()
            if hasattr(row, col) and pd.notna(getattr(row, col))
        }
        ours = G.judge(
            thresholds,
            age_group=str(getattr(row, M.AGE_GROUP_COL)),
            age=int(getattr(row, M.AGE_COL)),
            sex=str(getattr(row, M.SEX_COL)),
            measurements=values,
            body_ranges=body_ranges,
        ).grade
        theirs = str(getattr(row, M.GRADE_COL))
        if ours is None:
            undecidable += 1
        elif ours == theirs:
            agree += 1
        elif GRADE_ORDER.index(ours) < GRADE_ORDER.index(theirs):
            disagree_generous += 1  # 우리가 더 높은 등급을 줬다 = 후하다
        else:
            disagree_strict += 1
    decided = agree + disagree_generous + disagree_strict
    return {
        "n": int(len(graded)),
        "undecidable": undecidable,
        "decided": decided,
        "agree": agree,
        "generous": disagree_generous,
        "strict": disagree_strict,
        "agree_ratio": round(agree / decided, 4) if decided else 0.0,
    }


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
    body_ranges = _body_ranges(args)
    summary, dist, quantiles = build(df, thresholds)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # utf-8-sig — 검수하는 사람이 엑셀로 연다 (docs/02 §4)
    summary.to_csv(out / "age_band_score_summary.csv", index=False, encoding="utf-8-sig")
    dist.to_csv(out / "age_band_score_distribution.csv", index=False, encoding="utf-8-sig")
    quantiles.to_csv(out / "age_band_value_quantiles.csv", index=False, encoding="utf-8-sig")
    C.to_frame(thresholds).to_csv(out / "grade_thresholds.csv", index=False, encoding="utf-8-sig")
    if body_ranges:
        C.body_ranges_to_frame(body_ranges).to_csv(
            out / BODY_RANGES_FILE, index=False, encoding="utf-8-sig"
        )
    grades = build_grade_distribution(df, thresholds)
    grades.to_csv(out / "age_band_grade_distribution.csv", index=False, encoding="utf-8-sig")

    # 출력을 파일로 넘기면 로케일 인코딩을 쓴다. 한국어 윈도우(cp949)에 없는
    # 문자(⚠, em dash)를 넣지 않는다.
    skipped = int((summary["status"] != "ok").sum())
    if skipped:
        print(f"[알림] 눈금이 서지 않아 제외한 칸 {skipped}개. 사유는 요약 CSV 의 status 열")
    print(f"원자료 {len(df):,}행 · 문턱 {len(thresholds):,}건")
    print(f"요약 {len(summary):,}행 → {out / 'age_band_score_summary.csv'}")
    print(f"분포 {len(dist):,}행 → {out / 'age_band_score_distribution.csv'}")
    print(f"분위수 {len(quantiles):,}행 → {out / 'age_band_value_quantiles.csv'}")
    print(f"등급분포 {len(grades):,}행 → {out / 'age_band_grade_distribution.csv'}")

    report = concordance(df, thresholds, body_ranges)
    if report["n"]:
        print(
            f"\n[판정 대조] 표본 {report['n']:,}명 중 판정 불가 {report['undecidable']:,}"
            f" · 판정된 {report['decided']:,}명의 일치율 {report['agree_ratio']:.1%}"
        )
        print(f"  불일치 — 우리가 후한 쪽 {report['generous']:,} · 박한 쪽 {report['strict']:,}")
        # 기울기를 실측에서 읽는다. 어느 쪽으로 기우는지 미리 단정하지 않는다 —
        # 신체조성 누락은 후한 쪽으로, 결측을 판정 불가로 보는 규칙은 박한 쪽으로
        # 작용해서 방향이 상쇄된다 (docs/dev/AI-2 §5).
        lean = (
            "박한"
            if report["strict"] > report["generous"]
            else ("후한" if report["generous"] > report["strict"] else "양쪽 비슷한")
        )
        print(f"  {lean} 쪽으로 기운다 — 사유는 docs/dev/AI-2 §5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
