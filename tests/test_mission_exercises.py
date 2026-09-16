"""운동 표 — 합치는 근거가 데이터인지 본다 (docs/05 §3.1)."""

from __future__ import annotations

import pandas as pd

from family_fitness_ai.mission.exercises import (
    KIND_BUNDLE,
    KIND_SINGLE,
    build,
    candidates,
    co_occurring,
    phase_shares,
)


def _cells(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """`(단계, 이름들, 횟수들)`."""
    return pd.DataFrame(
        [
            {
                "chunk_id": f"c{i}",
                "phase": phase,
                "exercise_names": names,
                "exercise_counts": counts,
            }
            for i, (phase, names, counts) in enumerate(rows)
        ]
    )


def _vocabulary(rows: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exercise_name": name,
                "raw_forms": name,
                "count": count,
                "phases": "본운동",
                "age_groups": "성인",
                "fitness_factors": "",
            }
            for name, count in rows
        ]
    )


def test_같은_칸에_함께_처방되면_합치지_않는다() -> None:
    cells = _cells([("본운동", "하지 루틴 스트레칭1;하지 루틴 스트레칭2", "10;5")])
    together = co_occurring(cells)
    found = candidates(["하지 루틴 스트레칭1", "하지 루틴 스트레칭2"], together)

    assert found, "꼬리 규칙이 후보로 잡아야 한다"
    assert all(not c.merged for c in found)
    assert found[0].blocked == (("하지 루틴 스트레칭1", "하지 루틴 스트레칭2"),)


def test_함께_나오지_않으면_대표_이름으로_접는다() -> None:
    # 두 이름이 서로 다른 칸에만 나온다 — 막을 근거가 없다
    cells = _cells(
        [
            ("본운동", "골반 스트레칭;달리기", "10;1"),
            ("본운동", "골반 스트레칭2;걷기", "3;1"),
        ]
    )
    frame, found = build(
        _vocabulary([("골반 스트레칭", 10), ("골반 스트레칭2", 3), ("달리기", 1), ("걷기", 1)]),
        cells,
    )

    merged = [c for c in found if c.merged]
    assert len(merged) == 1
    names = frame["exercise_name"].tolist()
    assert "골반 스트레칭" in names, "많이 처방된 원문이 대표다"
    assert "골반 스트레칭2" not in names, "접힌 이름은 행을 내지 않는다"
    assert frame.loc[frame["exercise_name"] == "골반 스트레칭", "merge_reason"].item() == "꼬리"


def test_막힌_상대를_행에_남긴다() -> None:
    """막혔다는 기록이 없으면 다음 사람이 같은 후보를 다시 낸다."""
    cells = _cells([("본운동", "엉덩이 스트레칭;엉덩이 스트레칭2", "10;5")])
    frame, _ = build(_vocabulary([("엉덩이 스트레칭", 10), ("엉덩이 스트레칭2", 5)]), cells)

    row = frame.loc[frame["exercise_name"] == "엉덩이 스트레칭"].iloc[0]
    assert row["blocked_with"] == "엉덩이 스트레칭2"
    assert row["merge_reason"] == "", "합치지 않았으므로 근거가 없다"


def test_묶음_이름을_가른다() -> None:
    cells = _cells([("본운동", "맨몸운동  루틴프로그램;팔굽혀펴기", "10;5")])
    frame, _ = build(_vocabulary([("맨몸운동  루틴프로그램", 10), ("팔굽혀펴기", 5)]), cells)

    kinds = dict(zip(frame["exercise_name"], frame["kind"], strict=True))
    assert kinds["맨몸운동  루틴프로그램"] == KIND_BUNDLE
    assert kinds["팔굽혀펴기"] == KIND_SINGLE


def test_단계_비중은_횟수로_낸다() -> None:
    cells = _cells(
        [
            ("준비운동", "팔굽혀펴기", "30"),
            ("본운동", "팔굽혀펴기", "70"),
        ]
    )
    share = phase_shares(cells)["팔굽혀펴기"]

    assert share == {"준비운동": 0.3, "본운동": 0.7}
    assert abs(sum(share.values()) - 1.0) < 1e-9


def test_indoor_를_더미로_채우지_않는다() -> None:
    cells = _cells([("본운동", "팔굽혀펴기", "5")])
    frame, _ = build(_vocabulary([("팔굽혀펴기", 5)]), cells)

    assert (frame["indoor"] == "").all()
