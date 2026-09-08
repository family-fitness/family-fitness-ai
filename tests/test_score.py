"""점수 정규화 규약 (docs/02 §5.2·§5.3) 검사."""

import numpy as np
import pytest

from family_fitness_ai.stats.criteria import parse_age_band
from family_fitness_ai.stats.items import normalise_header
from family_fitness_ai.stats.score import build_anchors, score

RNG = np.random.default_rng(0)
REF = RNG.normal(40.0, 6.0, 5000)
TH = {3: 34.8, 2: 39.5, 1: 44.4}  # docs/02 §5.6 유소년 여 11세 상대악력


def _one(value: float, anchors, ref, *, lower_is_better: bool = False) -> float:
    return float(score(np.array([value]), anchors, ref, lower_is_better=lower_is_better)[0])


def test_문턱은_정확히_40_60_80이다() -> None:
    anchors, reason = build_anchors(REF, TH, lower_is_better=False)
    assert reason == "ok"
    assert [_one(TH[g], anchors, REF) for g in (3, 2, 1)] == [40.0, 60.0, 80.0]


def test_문서의_실측_예시를_재현한다() -> None:
    anchors, _ = build_anchors(REF, TH, lower_is_better=False)
    assert round(_one(41.3, anchors, REF), 1) == 67.6


def test_앵커_바깥은_0과_100으로_잘린다() -> None:
    anchors, _ = build_anchors(REF, TH, lower_is_better=False)
    assert _one(-999, anchors, REF) == 0.0
    assert _one(999, anchors, REF) == 100.0


def test_작을수록_우수한_항목도_점수는_클수록_좋다() -> None:
    ref = RNG.normal(10.0, 1.5, 5000)
    th = {3: 11.5, 2: 10.5, 1: 9.5}
    anchors, reason = build_anchors(ref, th, lower_is_better=True)
    assert reason == "ok"
    assert [_one(th[g], anchors, ref, lower_is_better=True) for g in (3, 2, 1)] == [
        40.0,
        60.0,
        80.0,
    ]
    fast, slow = (
        _one(8.0, anchors, ref, lower_is_better=True),
        _one(12.0, anchors, ref, lower_is_better=True),
    )
    assert fast > slow


def test_3등급_문턱이_없으면_p25로_채운다() -> None:
    anchors, reason = build_anchors(REF, {2: 39.5, 1: 44.4}, lower_is_better=False)
    assert reason == "ok"
    assert anchors.grade3_from_p25 is True
    assert anchors.x[1] == pytest.approx(float(np.percentile(REF, 25)))


def test_문턱이_겹치면_눈금을_만들지_않는다() -> None:
    # 유아기 윗몸말아올리기처럼 2·3등급이 둘 다 0인 칸
    anchors, reason = build_anchors(REF, {3: 0.0, 2: 0.0, 1: 44.4}, lower_is_better=False)
    assert anchors is None and reason == "anchors_not_monotonic"


def test_p25_대체값이_2등급을_넘으면_눈금을_만들지_않는다() -> None:
    # 성인 여 25~29 제자리멀리뛰기에서 실제로 일어난다 (docs/02 §6 ⑩)
    values = np.full(1000, 150.0)
    anchors, reason = build_anchors(values, {2: 144.0, 1: 156.0}, lower_is_better=False)
    assert anchors is None and reason == "anchors_not_monotonic"


def test_1_2등급이_없으면_점수를_내지_않는다() -> None:
    assert build_anchors(REF, {3: 34.8}, lower_is_better=False) == (None, "no_grade_1_2")


def test_표본이_적으면_선형으로_떨어진다() -> None:
    anchors, _ = build_anchors(REF[:10], TH, lower_is_better=False)
    assert anchors.method == "linear"


def test_점수는_단조증가한다() -> None:
    anchors, _ = build_anchors(REF, TH, lower_is_better=False)
    xs = np.linspace(20.0, 60.0, 200)
    ys = score(xs, anchors, REF, lower_is_better=False)
    assert np.all(np.diff(ys) >= -1e-9)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("48~53", (48, 53)),
        (11, (11, 11)),
        ("19~24", (19, 24)),
        ("85이상", (85, 120)),
        ("", None),
        (None, None),
    ],
)
def test_연령_밴드_파싱(raw: object, expected: tuple[int, int] | None) -> None:
    assert parse_age_band(raw) == expected


def test_기준표_머리글은_공백을_지워_맞춘다() -> None:
    assert normalise_header("15m 왕복\n오래달리기 (회)") == "15m왕복오래달리기(회)"


def test_문턱_표는_csv로_왕복한다(tmp_path) -> None:
    """zip 에 기준표가 없어도 커밋된 CSV 로 같은 문턱을 얻는다."""
    from family_fitness_ai.stats import criteria as C

    src = [
        C.Threshold("유소년", "F", 11, 11, "028", 1, 44.4),
        C.Threshold("유소년", "F", 11, 11, "028", 2, 39.5),
    ]
    path = tmp_path / "grade_thresholds.csv"
    C.to_frame(src).to_csv(path, index=False, encoding="utf-8-sig")
    assert set(C.load(path)) == set(src)


def test_틸드_경로와_안내_메시지(tmp_path, monkeypatch) -> None:
    """Makefile 이 인자를 따옴표로 넘겨 셸이 ~ 를 풀지 못한다. 코드가 푼다."""
    from family_fitness_ai.ingest.measurements import load_dir

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "받은자료").mkdir()
    with pytest.raises(FileNotFoundError, match="CSV가 없다"):
        load_dir("~/받은자료")

    with pytest.raises(FileNotFoundError, match="디렉터리가 없다"):
        load_dir("~/없는경로")

    nested = tmp_path / "풀린곳" / "kspo-measure"
    nested.mkdir(parents=True)
    (nested / "a.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="하위 디렉터리에 있다"):
        load_dir(tmp_path / "풀린곳")
