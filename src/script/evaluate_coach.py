"""코치 편성이 키울 요인을 얼마나 담는지 잰다.

    python src/script/evaluate_coach.py              # 표로 본다
    python src/script/evaluate_coach.py --json out.json

고친 뒤에 좋아졌는지를 말로 하지 않고 숫자로 보려고 둔다. 사람을 지어내지 않고 또래
분포(value_quantiles)에서 값을 고른다. 연령대 · 성별 · 요인마다, 그 요인 항목은 또래
15번째 백분위에, 나머지 항목은 65번째에 두어 그 요인이 가장 낮은 아이를 만든다.
그러면 편성은 그 요인을 「지금 키우기 좋은 영역」으로 고른다.

규칙 편성(LLM 없음)으로 한 주를 짜고 이것을 잰다.

    본운동 대상 요인    본운동 클립 가운데, 클립 라벨의 요인이 대상 요인인 몫
                        (편성 결과의 fitness_factor 는 라벨이 비면 대상 요인으로
                        채워지므로 쓰지 않는다 — AGENTS §6-5)
    후보 상위 대상 요인  코치(LLM)에게 넘기는 후보(catalog.pool)의 본운동 앞 20편 가운데
                        대상 요인 몫. 코치는 이 차례를 보고 고른다
    처방 동작          본운동 클립 가운데 처방 근거에 이름이 나온 동작의 몫
    전날 겹침          BE 처럼 하루씩 이레를 짜며 받은 영상을 recent_video_ids 로 넘길 때,
                        그날 영상 가운데 바로 전날에도 나온 몫
    이레 새 영상       이레 동안 본운동에 나온 서로 다른 영상 수(사람당 평균)

코치(LLM) 경로도 LLM 없이 잰다. 진짜 코치 대신, 후보 목록 앞에서부터 그 회에 아직 안 쓴
동작을 단계별 편수만큼 고르는 가짜 코치를 넣는다. 코치는 목록 앞쪽을 고르는 버릇이 있어
후보 차례가 편성에 어떻게 이어지는지 볼 수 있다.

    코치 대상 요인     코치 경로 편성의 본운동 가운데 대상 요인 몫
    채우기 교체        코치가 고른 것을 채우기(compose._fill)가 바꾼 편수(편성당 평균).
                        바꾼 만큼 코치 글이 규칙 문구로 바뀐다

LLM 을 부르지 않는다 — 부르려 하면 멈춘다. 처방 근거를 찾느라 임베딩은 쓴다(make embed-model).
    --without-lead     후보를 번갈아 세우지 않은 차례(catalog._lead 를 끈 것)로 잰다.
                        고치기 전과 견줄 때 쓴다
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

os.environ["COACH_LLM"] = "0"  # 이 도구는 규칙 편성만 잰다. 설정을 읽기 전에 끈다

from family_fitness_ai.coach import compose  # noqa: E402
from family_fitness_ai.coach import llm as coach_llm  # noqa: E402
from family_fitness_ai.common import llm  # noqa: E402
from family_fitness_ai.common.items import AGE_GROUP_ITEMS, ITEMS, age_group_of  # noqa: E402
from family_fitness_ai.stats import tables  # noqa: E402
from family_fitness_ai.video import catalog  # noqa: E402

#: 재는 사람들. 또래 분포가 있는 나이만 둔다(유소년은 11 · 12세만 있다 — AGENTS §6).
PEOPLE = (
    (60, "개월"),
    (11, "세"),
    (12, "세"),
    (15, "세"),
    (17, "세"),
    (35, "세"),
    (50, "세"),
)
SEXES = ("M", "F")
LOW, HIGH = 15, 65
START = date(2026, 10, 5)
TOP = 20
DAYS = 7


@dataclass
class Case:
    age: int
    unit: str
    sex: str
    factor: str
    measurements: dict[str, float]

    @property
    def age_group(self) -> str:
        return age_group_of(self.age, self.unit)

    @property
    def name(self) -> str:
        return f"{self.age_group} {self.age}{self.unit} {self.sex} {self.factor}"


def _value(age_group: str, sex: str, age: int, code: str, percentile: int) -> float | None:
    """또래의 그 백분위에 해당하는 값. 작을수록 좋은 항목은 뒤집어 고른다."""
    peer = tables.peer(age_group, sex, age, code)
    if peer is None or not peer.enough:
        return None
    item = ITEMS.get(code)
    at = 100 - percentile if item and item.lower_is_better else percentile
    return float(round(peer.quantiles[at], 2))


def cases() -> list[Case]:
    out: list[Case] = []
    for age, unit in PEOPLE:
        group = age_group_of(age, unit)
        codes = [code for code in AGE_GROUP_ITEMS[group] if code in ITEMS]
        factors = sorted({ITEMS[code].factor for code in codes})
        for sex in SEXES:
            for factor in factors:
                values: dict[str, float] = {}
                for code in codes:
                    low = ITEMS[code].factor == factor
                    value = _value(group, sex, age, code, LOW if low else HIGH)
                    if value is not None:
                        values[code] = value
                if any(ITEMS[code].factor == factor for code in values):
                    out.append(Case(age, unit, sex, factor, values))
    return out


def _labels() -> dict[tuple[str, int], catalog.Clip]:
    return {(clip.video_id, clip.start_sec): clip for clip in catalog.clips()}


def measure(case: Case, labels: dict[tuple[str, int], catalog.Clip]) -> dict[str, Any] | None:
    profile = compose.RunProfile(
        ref="eval",
        role="주행자",
        age=case.age,
        age_unit=case.unit,
        sex=case.sex,
        input_level="L2",
        measurements=case.measurements,
    )
    read = compose.read_profile(profile)
    if read.factor != case.factor:
        return None  # 값을 그렇게 두었는데도 다른 요인이 가장 낮다 — 재지 않는다
    constraints = compose.Constraints(days_per_week=3, minutes_per_session=20)
    plan = compose.build([profile], START, 1, constraints)
    if plan.proposal is None:
        return {"case": case.name, "refused": plan.refusal_reason}

    main = [
        session
        for mission in plan.proposal["missions"]
        for session in mission["sessions"]
        if session["phase"] == "본운동"
    ]
    clips = [labels.get((s["video"]["video_id"], s["video"]["start_sec"])) for s in main]
    prescribed = catalog.prescribed_names(read.chunks[:4])
    # develop 처럼 grow 칸이 없는 판과도 견줄 수 있게, 있을 때만 넘긴다.
    grow: dict[str, Any] = {}
    if "grow" in inspect.signature(catalog.pool).parameters:
        grow["grow"] = read.factor
    pool, _ = catalog.pool(
        case.age_group,
        conditions=constraints.conditions(),
        factor=read.factor,
        prescribed=prescribed,
        level=catalog.level_of(read.percentile),
        seed=START.isoformat(),
        **grow,
    )
    top = [clip for clip in pool if clip.phase == "본운동"][:TOP]
    overlap, shown, mains = _week_of_days(profile)
    coach_main, coach_target, swapped = _coach_week(profile, constraints, labels, case.factor)
    return {
        "case": case.name,
        "age_group": case.age_group,
        "factor": case.factor,
        "main": len(main),
        "main_target": sum(1 for c in clips if c and c.fitness_factor == case.factor),
        "main_prescribed": sum(1 for c in clips if c and c.exercise_name in prescribed),
        "top": len(top),
        "top_target": sum(1 for c in top if c.fitness_factor == case.factor),
        "day_overlap": overlap,
        "day_videos": shown,
        "week_main_videos": mains,
        "coach_main": coach_main,
        "coach_target": coach_target,
        "coach_swapped": swapped,
    }


def _top_picks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """후보 목록 앞에서부터 고르는 가짜 코치.

    한 주 안에서 같은 동작, 한 회 안에서 같은 영상은 피한다.
    """
    week: set[str] = set()
    days = []
    for slot in payload["자리"]:
        videos: set[str] = set()
        rows = []
        for phase, want in slot["단계별_편수"].items():
            picked = 0
            for clip in payload["클립"]:
                if picked >= want:
                    break
                if clip["단계"] != phase or clip["이름"] in week or clip["영상"] in videos:
                    continue
                rows.append({"id": clip["id"], "phase": phase})
                week.add(clip["이름"])
                videos.add(clip["영상"])
                picked += 1
        days.append(
            {
                "day_offset": slot["day_offset"],
                "title": "",
                "child": "",
                "parent": "",
                "reason": "",
                "clips": rows,
            }
        )
    return days


def _no_llm(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("평가 도구는 LLM 을 부르지 않는다")


def _coach_week(
    profile: compose.RunProfile,
    constraints: compose.Constraints,
    labels: dict[tuple[str, int], catalog.Clip],
    factor: str,
) -> tuple[int, int, int]:
    """가짜 코치로 한 주를 짠다. (본운동 편수, 그중 대상 요인, 채우기가 바꾼 편수)."""
    saved = coach_llm.enabled, coach_llm.plan_week, llm.ask_json_or_none
    coach_llm.enabled = lambda: True
    coach_llm.plan_week = _top_picks
    llm.ask_json_or_none = _no_llm
    try:
        plan = compose.build([profile], START, 1, constraints)
    finally:
        coach_llm.enabled, coach_llm.plan_week, llm.ask_json_or_none = saved
    if plan.proposal is None:
        return 0, 0, 0
    main = [
        labels.get((s["video"]["video_id"], s["video"]["start_sec"]))
        for mission in plan.proposal["missions"]
        for s in mission["sessions"]
        if s["phase"] == "본운동"
    ]
    filled = re.search(r"목록에서 (\d+)편 채움", plan.steps[2].summary)
    return (
        len(main),
        sum(1 for clip in main if clip and clip.fitness_factor == factor),
        int(filled.group(1)) if filled else 0,
    )


def _week_of_days(profile: compose.RunProfile) -> tuple[int, int, int]:
    """BE 처럼 하루씩 이레를 짠다.

    (전날과 겹친 영상 수, 둘째 날부터 나온 영상 수, 본운동 영상 가짓수)를 돌려준다.
    """
    recent: list[str] = []
    yesterday: set[str] = set()
    overlap = shown = 0
    main_videos: set[str] = set()
    for day in range(DAYS):
        constraints = compose.Constraints(
            days_per_week=1, minutes_per_session=20, recent_video_ids=tuple(recent[:150])
        )
        plan = compose.build([profile], START + timedelta(days=day), 1, constraints)
        if plan.proposal is None:
            continue
        sessions = [s for m in plan.proposal["missions"] for s in m["sessions"]]
        today = [s["video"]["video_id"] for s in sessions]
        main_videos |= {s["video"]["video_id"] for s in sessions if s["phase"] == "본운동"}
        if day:
            overlap += sum(1 for video in today if video in yesterday)
            shown += len(today)
        yesterday = set(today)
        # 최근 것부터. 같은 영상은 한 번만.
        recent = list(dict.fromkeys([*reversed(today), *recent]))
    return overlap, shown, len(main_videos)


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "-"


def report(rows: list[dict[str, Any]]) -> str:
    measured = [row for row in rows if "main" in row]
    refused = [row for row in rows if "refused" in row]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        groups[row["age_group"]].append(row)
        groups["전체"].append(row)

    lines = [
        f"{'':8} {'사람':>4} {'본운동 대상 요인':>14} {'후보 상위 대상 요인':>16} {'처방 동작':>9}"
        f" {'전날 겹침':>9} {'이레 새 영상':>10} {'코치 대상 요인':>12} {'채우기 교체':>10}",
    ]
    for name in [*sorted(k for k in groups if k != "전체"), "전체"]:
        part = groups[name]
        main = sum(r["main"] for r in part)
        top = sum(r["top"] for r in part)
        coach_main = sum(r["coach_main"] for r in part)
        lines.append(
            f"{name:8} {len(part):>4} "
            f"{_share(sum(r['main_target'] for r in part), main):>14} "
            f"{_share(sum(r['top_target'] for r in part), top):>16} "
            f"{_share(sum(r['main_prescribed'] for r in part), main):>9}"
            f" {_share(sum(r['day_overlap'] for r in part), sum(r['day_videos'] for r in part)):>9}"
            f" {sum(r['week_main_videos'] for r in part) / len(part):>10.1f}"
            f" {_share(sum(r['coach_target'] for r in part), coach_main):>12}"
            f" {sum(r['coach_swapped'] for r in part) / len(part):>10.1f}"
        )
    by_factor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        by_factor[row["factor"]].append(row)
    lines.append("")
    lines.append("요인별 본운동 대상 요인 몫")
    for factor, part in sorted(by_factor.items()):
        main = sum(r["main"] for r in part)
        lines.append(f"  {factor:6} {_share(sum(r['main_target'] for r in part), main):>5}")
    if refused:
        lines.append(f"\n편성 거부 {len(refused)}건: " + ", ".join(r["case"] for r in refused))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="코치 편성이 키울 요인을 얼마나 담는지 잰다")
    parser.add_argument("--json", type=Path, help="사람마다의 결과를 이 파일에 쓴다")
    parser.add_argument(
        "--without-lead", action="store_true", help="후보를 번갈아 세우지 않은 차례로 잰다"
    )
    args = parser.parse_args()
    if args.without_lead and hasattr(catalog, "_lead"):
        catalog._lead = lambda ranked, grow, recent: ranked

    labels = _labels()
    rows = [row for row in (measure(case, labels) for case in cases()) if row is not None]
    print(report(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
