"""요인 단위 공통 척도 — 준거참조 정규화.

규약은 docs/02 §5.2·§5.3. 등급 문턱을 공통 눈금(3등급 40 · 2등급 60 · 1등급 80)에
고정하고 그 사이를 단조 보간한다. 점수는 언제나 클수록 좋다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SAMPLE = 30  # 이보다 적으면 ECDF를 믿지 않고 선형으로 떨어뜨린다


@dataclass(frozen=True)
class Anchors:
    """점수 눈금. x 는 언제나 '클수록 좋다' 공간의 값이고, y 는 그 점수다.

    앵커 수는 기준표가 정의한 문턱 수를 따른다. 운동체력(순발력·민첩성·협응력)은
    3등급 문턱이 없어 40점 자리가 비고, 0에서 60까지가 한 구간이 된다.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    method: str  # "ecdf" | "linear"

    @property
    def has_grade3(self) -> bool:
        return 40.0 in self.y


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

    # 기준표에 있는 문턱만 앵커로 쓴다. 3등급 문턱이 없는 것은 결측이 아니라
    # 그 항목이 3등급 판정 대상이 아니라는 뜻이다 (docs/02 §5.2).
    inner = [(t[3], 40.0)] if 3 in t else []
    inner += [(t[2], 60.0), (t[1], 80.0)]

    # 0·100 앵커는 분포에서 온다. 문턱 앵커를 뒤집지 않도록 바깥으로만 넓힌다.
    x_low = min(float(np.percentile(v, 2)), inner[0][0])
    x_high = max(float(np.percentile(v, 98)), t[1])

    x = (x_low, *(xv for xv, _ in inner), x_high)
    y = (0.0, *(yv for _, yv in inner), 100.0)
    if not all(a < b for a, b in zip(x, x[1:], strict=False)):
        # 문턱이 겹치거나 뒤집히면 눈금이 성립하지 않는다. 유아기 윗몸말아올리기처럼
        # 2·3등급이 둘 다 0회인 칸이 여기 걸린다 — 40점과 60점을 가를 값이 없다.
        return None, "anchors_not_monotonic"

    method = "ecdf" if v.size >= MIN_SAMPLE else "linear"
    return Anchors(x=x, y=y, method=method), "ok"


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
