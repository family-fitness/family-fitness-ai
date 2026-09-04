"""요인 단위 공통 척도 — 준거참조 정규화.

규약은 docs/02 §5.2·§5.3. 등급 문턱을 공통 눈금(3등급 40 · 2등급 60 · 1등급 80)에
고정하고 그 사이를 단조 보간한다. 점수는 언제나 클수록 좋다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANCHOR_SCORES = (0.0, 40.0, 60.0, 80.0, 100.0)
MIN_SAMPLE = 30  # 이보다 적으면 ECDF를 믿지 않고 선형으로 떨어뜨린다


@dataclass(frozen=True)
class Anchors:
    """점수 눈금. x 는 언제나 '클수록 좋다' 공간의 값이다."""

    x: tuple[float, float, float, float, float]
    grade3_from_p25: bool  # 3등급 문턱이 없어 p25로 채웠는가
    method: str  # "ecdf" | "linear"

    @property
    def y(self) -> tuple[float, ...]:
        return ANCHOR_SCORES


def build_anchors(
    values: np.ndarray,
    thresholds: dict[int, float],
    *,
    lower_is_better: bool,
) -> tuple[Anchors | None, str]:
    """분포와 등급 문턱으로 앵커를 만든다.

    만들지 못하면 사유를 함께 돌려준다. 눈금이 서지 않는 칸을 조용히 버리지 않기
    위해서다 — 빠진 칸은 산출물에 사유와 함께 남는다.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None, "no_values"

    sign = -1.0 if lower_is_better else 1.0
    v = sign * v
    t = {g: sign * x for g, x in thresholds.items()}

    if 1 not in t or 2 not in t:
        return None, "no_grade_1_2"

    # 3등급 문턱이 없는 항목은 40점 앵커를 그 칸의 p25 로 채운다 (docs/02 §5.3).
    grade3_from_p25 = 3 not in t
    t3 = t.get(3, float(np.percentile(v, 25)))

    # 0·100 앵커는 분포에서 온다. 문턱 앵커를 뒤집지 않도록 바깥으로만 넓힌다.
    x_low = min(float(np.percentile(v, 2)), t3)
    x_high = max(float(np.percentile(v, 98)), t[1])

    x = (x_low, t3, t[2], t[1], x_high)
    if not all(a < b for a, b in zip(x, x[1:], strict=False)):
        # 문턱이 겹치거나 뒤집히면 눈금이 성립하지 않는다. 유아기 윗몸말아올리기처럼
        # 2·3등급이 둘 다 0회인 칸이 여기 걸린다 — 40점과 60점을 가를 값이 없다.
        return None, "anchors_not_monotonic"

    method = "ecdf" if v.size >= MIN_SAMPLE else "linear"
    return Anchors(x=x, grade3_from_p25=grade3_from_p25, method=method), "ok"


def score(
    values: np.ndarray,
    anchors: Anchors,
    reference: np.ndarray,
    *,
    lower_is_better: bool,
) -> np.ndarray:
    """값 → 0~100 점수.

    `reference` 는 그 칸의 분포다. ECDF 보간은 이 분포를 쓴다 — 사람이 몰린 구간에서
    변별력이 생긴다 (docs/02 §5.3).
    """
    sign = -1.0 if lower_is_better else 1.0
    v = sign * np.asarray(values, dtype=float)

    ref = np.sort(sign * np.asarray(reference, dtype=float)[np.isfinite(reference)])
    use_ecdf = anchors.method == "ecdf" and ref.size > 0

    def cdf(a: np.ndarray) -> np.ndarray:
        return np.searchsorted(ref, a, side="right") / ref.size

    xs, ys = np.asarray(anchors.x), np.asarray(anchors.y)
    out = np.full(v.shape, np.nan)
    finite = np.isfinite(v)

    # 구간은 (lo, hi] 로 잡는다. 문턱에 정확히 걸린 값이 그 문턱 점수를 받아야 한다.
    for i in range(len(xs) - 1):
        lo, hi, y_lo, y_hi = xs[i], xs[i + 1], ys[i], ys[i + 1]
        seg = finite & (v > lo) & (v <= hi)
        if not seg.any():
            continue
        if use_ecdf:
            f_lo, f_hi = cdf(np.array([lo, hi]))
            # 그 구간에 표본이 없으면(F 가 평평하면) 선형으로 떨어뜨린다
            w = (cdf(v[seg]) - f_lo) / (f_hi - f_lo) if f_hi > f_lo else (v[seg] - lo) / (hi - lo)
        else:
            w = (v[seg] - lo) / (hi - lo)
        out[seg] = y_lo + (y_hi - y_lo) * np.clip(w, 0.0, 1.0)

    out[finite & (v <= xs[0])] = 0.0
    out[finite & (v >= xs[-1])] = 100.0
    return out
