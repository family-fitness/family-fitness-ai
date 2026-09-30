"""국민체력100 동영상(공공데이터 API) → data/release/kspo_videos.csv.

    python -m family_fitness_ai.video.kspo           받아 둔 원자료로 표를 만든다
    python -m family_fitness_ai.video.kspo --fetch   API 에서 다시 받고 링크도 다시 본다

유튜브 영상은 한 편에 동작이 스무 개라 화면 글자로 끊어 클립을 만든다. 공단에는
처음부터 한 동작을 담은 0~2분대 영상이 있다 — **한 편이 그대로 클립 하나**다. 끊지
않는다. 서버가 주고받는 모양을 지키려고 시작은 0, 끝과 길이는 영상 길이로 둔다.
`catalog.clips()` 가 이 표를 유튜브 클립과 한 목록으로 합친다. 그 뒤로는 어느 쪽인지
가리지 않고 똑같이 고른다. 칸은 video_clips.csv 와 clip_labels.csv 를 한 줄에 이은
모양에 공단 영상에만 있는 것을 더했다.

API 는 영상을 여섯 갈래로 나눠 주고, 한 영상은 한 갈래에만 든다(동영상 목록 조회가
그 색인이다). 클립으로 쓰는 것은 운동처방가이드와 운동처방동영상 둘이다.
생애주기별표준운동·목적별루틴은 모두 5분이 넘어 영상은 쓰지 않고, 운동명으로
메타데이터(체력요인·단계)만 빌린다. 근골격계운동은 질환자 재활, 체력인증측정방법은
측정 요령이라 받지 않는다.

메타데이터는 API 에서 온다
    체력요인·추천 체력수준  가이드는 제 칸(ftns_fctr_nm · ftns_lvl_nm)에 있다. 처방동영상은
                            그 칸이 없어, 같은 운동명의 가이드(없으면 루틴)가 준 값을 쓴다.
                            설명에 「유산소운동에 해당하는」처럼 갈래가 적혀 있으면 요인은
                            그 갈래가 먼저다(KINDS) — 「운동프로그램」 묶음은 요인 칸이
                            모두 「유연성」으로 잘못 온다.
    단계                    짧은 영상 조회에는 단계 칸이 없다. 같은 운동명이 표준운동·루틴에서
                            받은 단계(「준비 운동」…)를 쓰고, 없으면 체력요인이 유연성이면
                            준비·정리, 아니면 본운동. 요인도 없으면 이름으로 본다.
    도구·장소·인원·세트     영상의 장면 줄에 적힌 값 중 가장 많은 것.
    조용한지               API 에 없다. 유튜브 클립과 같은 규칙(이름)으로 본다.

가려 쓰는 것
    5분 이상, 제목·이름에 「루틴」(여러 동작을 묶은 루틴 프로그램), 질환·부상용(오십견·
    요통·경부통·발목염좌 … 과 「…예방」 — 진료 쪽), 「짝」 운동과 둘 이상이 하는 것,
    헬스장·수영장·운동장, 헬스기구·바벨, 어르신, 파일이 없는 영상은 뺀다. 짧은 영상
    안에서 변형이 이어지는 것은 한 동작으로 보고 제목을 이름으로 쓴다. 「공통」은
    청소년·성인에 둔다.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from family_fitness_ai.common.settings import settings
from family_fitness_ai.rag.medical import is_medical
from family_fitness_ai.video import vocabulary
from family_fitness_ai.video.labels import _rules

API = "https://apis.data.go.kr/B551014/SRVC_TODZ_VDO_PKG/"
GUIDE = "TODZ_VDO_TRNG_GUIDE_I"
VIDEO_OP = "TODZ_VDO_TRNG_VIDEO_I"
STD = "TODZ_VDO_STD_FTNS_I"
ROUTINE = "TODZ_VDO_ROUTINE_I"
#: 클립으로 쓰는 조회와 인용에 붙일 이름.
CLIP_OPS = {VIDEO_OP: "운동처방동영상", GUIDE: "운동처방가이드"}
#: 받아 두는 조회. 표준운동·루틴은 운동명으로 메타데이터만 빌린다.
OPS = {**CLIP_OPS, STD: "생애주기별표준운동", ROUTINE: "목적별루틴운동"}
#: 장면마다 단계를 적어 주는 칸. 표준운동은 앞, 루틴은 뒤다.
PHASE_FIELDS = ("trng_sqnc_nm", "trng_se_nm")
PHASES = ("준비운동", "본운동", "정리운동")
#: API 는 http 로 적어 주지만 http 는 https 로 301 한다. 앱이 http 를 막으니 https 로 낸다.
VIDEO = "https://openapi.kspo.or.kr/web/video/"
RAW = "kspo"
OUT = "kspo_videos.csv"

#: 이보다 짧아야 쓴다(5분 미만). 긴 것은 여러 동작을 이은 프로그램이다.
MAX_SEC = 300
#: 우리 연령대로. 「공통」은 청소년·성인에 둔다 — 유소년에게는 어른 동작이 섞인다.
#: 어르신은 범위 밖이고, 유아기는 파일이 없다.
AGES = {"유소년": ("유소년",), "청소년": ("청소년",), "성인": ("성인",), "공통": ("청소년", "성인")}
#: 공단의 체력요인 이름 → 우리 여덟. 둘을 묶어 적은 것은 둘 다다.
FACTORS = {
    "유연성": ("유연성",),
    "근력": ("근력",),
    "근력/근지구력": ("근력", "근지구력"),
    "심폐지구력": ("심폐지구력",),
    "전신지구력": ("심폐지구력",),
    "유산소": ("심폐지구력",),
    "순발력": ("순발력",),
    "민첩성": ("민첩성",),
    "민첩성/순발력": ("민첩성", "순발력"),
    "민첩성/성능": ("민첩성",),
    "협응성": ("협응력",),
    "협응력": ("협응력",),
    "평형성": ("평형성",),
}
#: 운동처방가이드 설명 「… 중, 가슴운동에 해당하는 …」의 갈래 → 우리 요인. 「운동프로그램」
#: 「체력 증진 운동프로그램」 묶음은 API 가 체력요인을 모두 「유연성」으로 적어, 빠르게
#: 걷기가 준비·정리 유연성으로, 팔굽혀펴기가 유연성으로 실렸다. 갈래가 적혀 있으면 API
#: 체력요인보다 먼저 쓴다. 받아 둔 응답에 나온 갈래뿐이다 — 없는 갈래는 API 요인을 쓴다.
KINDS = {
    "유산소운동": ("심폐지구력",),
    "가슴운동": ("근력",),
    "등운동": ("근력",),
    "몸통운동": ("근력",),
    "팔/어깨운동": ("근력",),
    "하체운동": ("근력",),
    "스트레칭": ("유연성",),
}
_KIND = re.compile(r"중,\s*(\S+?)에 해당하는")
#: 제목·설명에 이 말이 들면 질환·부상용이다 — 진료 쪽이라 쓰지 않는다. 질문 가리기
#: (rag.medical)가 「염좌」만 잡아, 가이드에 섞인 오십견·경부통·요통 영상을 여기서 더 잡는다.
CONDITIONS = ("예방", "요통", "경부통", "오십견", "염좌", "부동증후군", "질환", "재활", "통증")
#: 집에서 못 하는 곳.
AWAY = ("헬스장", "수영장", "운동장")
#: 집에 없는 기구.
GYM = (
    "헬스기구", "바벨", "원판", "벤치", "바", "실내자전거", "자전거", "트레드밀",
    "스텝박스", "스텝퍼", "스탭퍼", "킥판", "아쿠아봉", "폼롤러",
)  # fmt: skip
#: 집에 있는 것 — 도구로 치지 않는다.
HOUSEHOLD = ("", "매트", "의자", "수건", "수건(낮은 쿠션)", "테이블", "소파", "베개", "물병")
#: 좁은 데서는 못 하는 것.
ROOMY = ("사다리", "줄사다리", "콘", "라바콘", "하프콘", "계단", "배드민턴 라켓")
#: 체력수준을 이름으로 적은 것. 숫자는 1(낮음)~5(높음)이다 — 청소년 「1~2」에는
#: 사이드 스텝, 「4~5」에는 순간반응 콘 찍기가 있다. 명세에는 없고 자료를 보고 읽었다.
LEVEL_WORDS = {"초급": (1, 2), "중급": (2, 4), "고급": (4, 5)}

COLUMNS = (
    # video_clips.csv 와 같은 칸 — 한 편이 한 클립이라 seq 는 1, 시작은 0 이다.
    "video_id", "seq", "name_on_video", "phase_on_video", "start_sec", "end_sec", "duration_sec",
    # clip_labels.csv 와 같은 칸
    "exercise_name", "fitness_factor", "phase", "is_exercise", "home_ok", "quiet",
    "needs_props", "source", "score",
    # 공단 영상에만 있는 칸
    "age_group", "level_lo", "level_hi", "sets", "reps", "hold", "url", "citation_label",
)  # fmt: skip


def levels(text: str) -> tuple[int, int]:
    """「3~5」 → (3, 5). 적혀 있지 않거나 「공통」이면 누구나다."""
    text = (text or "").strip()
    if text in LEVEL_WORDS:
        return LEVEL_WORDS[text]
    numbers = [int(n) for n in re.findall(r"[1-5]", text)]
    return (min(numbers), max(numbers)) if numbers else (1, 5)


def api_phase(text: str) -> str:
    """「본 운동(순환식 근력 운동)」 → 본운동. 단계가 아닌 구분(「스트레칭」·「공통」)은 빈칸."""
    word = re.sub(r"\s|\(.*?\)", "", text or "")
    return word if word in PHASES else ""


def _row_phase(row: dict[str, Any]) -> str:
    return next((p for f in PHASE_FIELDS if (p := api_phase(row.get(f) or ""))), "")


def kind_factors(description: str) -> tuple[str, ...]:
    """설명이 적은 갈래의 요인. 갈래가 없거나 모르는 갈래면 빈 채다."""
    kind = _KIND.search(description or "")
    return KINDS.get(kind[1], ()) if kind else ()


def rule_phases(name: str, factors: tuple[str, ...]) -> tuple[str, ...]:
    """API 가 단계를 모를 때. 유연성 동작은 준비·정리 둘 다, 나머지는 본운동.
    요인도 없으면 이름으로 본다(유튜브 클립과 같은 규칙)."""
    if "유연성" in factors or (not factors and _rules(name, "")["phase"] == "준비운동"):
        return ("준비운동", "정리운동")
    return ("본운동",)


#: 단계 번호 「-1」·「-2」.
_STEP = re.compile(r"\s*-\s*\d+$")


def display(name: str) -> str:
    """화면에 나갈 이름. 단계 번호만 뗀다. 괄호는 둔다 — 「(Child's pose)」와
    「(Down dog pose)」처럼 괄호가 다른 동작을 가른다."""
    return _STEP.sub("", name or "").strip()


def join_key(name: str) -> str:
    """다른 조회의 같은 동작을 찾는 열쇠. 괄호·단계 번호를 뗀다 — 처방동영상
    「팔 굽혀 펴기」와 가이드 「팔 굽혀 펴기(매트)」, 표준운동 「목 스트레칭(Neck
    stretch)」과 가이드 「목 스트레칭」이 같다."""
    return vocabulary.normalized(re.sub(r"\(.*?\)|\d+$", "", display(name)))


@lru_cache
def _vocabulary() -> dict[str, str]:
    """처방 어휘를 잇는 열쇠 → 처방 어휘."""
    out: dict[str, str] = {}
    for name in vocabulary.all_names():
        out.setdefault(vocabulary.normalized(name), name)
        out.setdefault(join_key(name), name)
    return out


def exercise_name(name: str) -> str:
    """처방 어휘로 옮긴 이름. 편성이 「또래 처방에 나온 동작」을 앞세우는 데 쓴다."""
    table = _vocabulary()
    return table.get(vocabulary.normalized(name)) or table.get(join_key(name)) or ""


def _most(rows: list[dict[str, Any]], key: str) -> str:
    """장면 줄 가운데 가장 많이 적힌 값. 같은 수면 글자순 — 차례에 따라 바뀌지 않게."""
    values = collections.Counter(v for r in rows if (v := (r.get(key) or "").strip()))
    return sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if values else ""


# ── 운동명으로 빌리는 메타데이터 ───────────────────────────────────────────


@dataclass(frozen=True)
class Borrowed:
    """다른 조회가 같은 운동명에 붙인 값. 공단 원문 그대로 둔다(요인만 설명의 갈래가 먼저)."""

    factor: str
    level: str
    phases: tuple[str, ...]


def borrowed_from(rows_by_op: dict[str, list[dict[str, Any]]]) -> dict[str, Borrowed]:
    """운동명 → API 가 그 동작에 붙인 체력요인·체력수준·단계.

    한 영상은 한 조회에만 들어서 영상으로는 이을 수 없다. 운동명으로 잇는다.
    요인은 가이드가 먼저고 없으면 루틴, 수준은 가이드, 단계는 표준운동·루틴이다.
    가이드 설명에 갈래(KINDS)가 적혀 있으면 요인은 원문 대신 그 갈래의 요인이다.
    """
    guide_factor: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    routine_factor: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    level: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    phases: dict[str, set[str]] = collections.defaultdict(set)
    for op, rows in rows_by_op.items():
        for row in rows:
            key = join_key(row.get("trng_nm") or "")
            if not key:
                continue
            factor = (row.get("ftns_fctr_nm") or "").strip()
            if op == GUIDE and (kind := kind_factors(row.get("vdo_desc") or "")):
                # 가이드 설명에 갈래가 있으면 _video 와 같이 그쪽이 먼저다. 그래야 빌려
                # 가는 처방동영상(빠르게 걷기 00182)이 가이드(00601)와 어긋나지 않는다.
                factor = kind[0]
            if factor and op == GUIDE:
                guide_factor[key][factor] += 1
            elif factor and op == ROUTINE:
                routine_factor[key][factor] += 1
            if op == GUIDE and (row.get("ftns_lvl_nm") or "").strip():
                level[key][row["ftns_lvl_nm"].strip()] += 1
            if phase := _row_phase(row):
                phases[key].add(phase)

    def top(counts: collections.Counter[str] | None) -> str:
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if counts else ""

    keys = set(guide_factor) | set(routine_factor) | set(level) | set(phases)
    return {
        key: Borrowed(
            factor=top(guide_factor.get(key)) or top(routine_factor.get(key)),
            level=top(level.get(key)),
            phases=tuple(p for p in PHASES if p in phases.get(key, set())),
        )
        for key in sorted(keys)
    }


# ── 표 ─────────────────────────────────────────────────────────────────────


def build(
    rows_by_op: dict[str, list[dict[str, Any]]],
    alive: dict[str, bool],
    borrowed: dict[str, Borrowed] | None = None,
) -> list[dict[str, str]]:
    """API 원자료 → 표 한 장. 한 편이 한 클립이고, 연령대·요인·단계마다 한 줄이 된다."""
    borrowed = borrowed_from(rows_by_op) if borrowed is None else borrowed
    out: list[dict[str, str]] = []
    for op, kind in CLIP_OPS.items():
        videos: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows_by_op.get(op, []):
            videos[row["file_nm"]].append(row)
        for file_nm in sorted(videos):
            out += _video(videos[file_nm], kind, alive.get(file_nm, False), borrowed)
    return sorted(out, key=_order)


def _order(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["video_id"], row["age_group"], row["fitness_factor"], row["phase"]


def _video(
    scenes: list[dict[str, Any]], kind: str, alive: bool, borrowed: dict[str, Borrowed]
) -> list[dict[str, str]]:
    """영상 하나 → 클립 하나(를 연령대·요인·단계마다 한 줄씩). 쓰지 않는 영상이면 빈 목록."""
    file_nm = scenes[0]["file_nm"]
    title = _most(scenes, "vdo_ttl_nm")
    seconds = round(max(float(r.get("vdo_len") or 0) for r in scenes))
    ages = AGES.get(_most(scenes, "aggrp_nm"), ())
    about = " ".join((title, _most(scenes, "vdo_desc")))
    moves = {join_key(r.get("trng_nm") or "") for r in scenes} - {""}
    # 한 편이 한 클립이다. 장면마다 운동명이 같으면 그 이름, 변형이 이어지면(「앞으로·
    # 뒤로 당기기」, 「양발·한발 넘기기」) 영상 제목, 운동명이 아예 없어도 제목이다.
    name = display(_most(scenes, "trng_nm") if len(moves) == 1 else title)
    tool = _most(scenes, "tool_nm")
    place = _most(scenes, "trng_plc_nm")
    ruled = _rules(name, "")
    if (
        not alive
        or not ages
        or not 0 < seconds < MAX_SEC
        or re.search("루[틴팀]", f"{title} {name}")  # 루틴 프로그램 — 여러 동작을 묶었다
        or is_medical(about)
        or any(word in about for word in CONDITIONS)
        or "짝" in about  # 둘이 해야 한다. 처방동영상에는 인원 칸이 없다
        or _most(scenes, "nope_nm") not in ("", "1인 이상")
        or place in AWAY
        or tool in GYM
        or not ruled["is_exercise"]
    ):
        return []

    got = borrowed.get(join_key(name)) or borrowed.get(join_key(_most(scenes, "trng_nm")))
    factors = kind_factors(_most(scenes, "vdo_desc")) or FACTORS.get(
        _most(scenes, "ftns_fctr_nm") or (got.factor if got else ""), ()
    )
    lo, hi = levels(_most(scenes, "ftns_lvl_nm") or (got.level if got else ""))
    phases = (got.phases if got else ()) or rule_phases(name, factors)
    linked = exercise_name(name)
    home_ok = (
        bool(ruled["home_ok"])
        and tool not in ROOMY
        and ("실내" in place or not place)  # 「실외」뿐이면 좁은 데서 못 한다
    )
    clip = {
        "video_id": Path(file_nm).stem,
        "seq": "1",
        "name_on_video": name,
        "phase_on_video": "",  # 짧은 영상 조회에는 단계 칸이 없다
        "start_sec": "0",
        "end_sec": str(seconds),
        "duration_sec": str(seconds),
        "exercise_name": linked,
        "is_exercise": "True",
        "home_ok": str(home_ok),
        "quiet": str(bool(ruled["quiet"])),
        "needs_props": str(tool not in HOUSEHOLD),
        "source": "kspo",
        "score": "1.0" if linked else "0.0",
        "level_lo": str(lo),
        "level_hi": str(hi),
        "sets": _most(scenes, "set_cnt_nm"),
        "reps": _most(scenes, "rptt_tcnt_nm"),
        "hold": _most(scenes, "trng_hr_nm"),
        "url": VIDEO + file_nm,
        "citation_label": f"국민체력100 {kind} · {title or name}",
    }
    rows = []
    for age in ages:
        for factor in factors or ("",):
            for phase in phases:
                row = {**clip, "age_group": age, "fitness_factor": factor, "phase": phase}
                rows.append({column: row[column] for column in COLUMNS})
    return rows


# ── 받기 ────────────────────────────────────────────────────────────────────


def fetch(op: str, key: str) -> list[dict[str, Any]]:
    """조회 하나를 끝까지 받는다. 한 번에 천 줄씩.

    키는 serviceKey 로 요청 주소에 들어간다. httpx 는 요청마다 INFO 로 주소를 통째로
    남기고, raise_for_status 의 오류 문구에도 주소를 적는다. 로깅이 켜진 곳에서 받으면
    키가 로그에 남으므로, 받는 동안 httpx 로그를 WARNING 으로 낮추고 오류는 주소 없이
    다시 낸다.
    """
    quiet = logging.getLogger("httpx")
    before = quiet.level
    quiet.setLevel(logging.WARNING)
    try:
        return _fetch(op, key)
    finally:
        quiet.setLevel(before)


def _fetch(op: str, key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page, total = 1, None
    while total is None or len(rows) < total:
        try:
            response = httpx.get(
                API + op,
                params={
                    "serviceKey": key,
                    "pageNo": page,
                    "numOfRows": 1000,
                    "resultType": "json",
                },
                timeout=60,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            # 원래 오류는 주소(키 포함)를 문구에 담는다. 이어 붙이지 않는다.
            raise RuntimeError(
                f"{OPS.get(op, op)} {page}쪽 받기 실패 — HTTP {error.response.status_code}"
            ) from None
        except httpx.HTTPError as error:
            raise RuntimeError(
                f"{OPS.get(op, op)} {page}쪽 받기 실패 — {type(error).__name__}"
            ) from None
        body = response.json()["response"]["body"]
        total = int(body["totalCount"])
        items = body["items"]
        got = items["item"] if isinstance(items, dict) else items
        rows += got if isinstance(got, list) else [got]
        page += 1
        if page > math.ceil(total / 1000) + 1:  # 줄 수가 모자라게 와도 끝은 난다
            break
        time.sleep(0.3)
    return rows


def check_links(files: list[str]) -> dict[str, bool]:
    """영상 파일이 열리나. 없으면 공단 서버가 404 대신 /error.html 로 넘긴다."""
    client = httpx.Client(timeout=20, follow_redirects=True)

    def alive(file_nm: str) -> bool:
        try:
            response = client.head(VIDEO + file_nm)
        except httpx.HTTPError:
            return False
        kind = response.headers.get("content-type", "")
        return response.status_code == 200 and "html" not in kind

    with ThreadPoolExecutor(4) as pool:
        return dict(zip(files, pool.map(alive, files), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="API 에서 다시 받고 링크도 다시 본다")
    args = parser.parse_args()

    raw = settings().raw_dir / RAW
    raw.mkdir(parents=True, exist_ok=True)
    rows_by_op: dict[str, list[dict[str, Any]]] = {}
    for op in OPS:
        path = raw / f"{op}.json"
        if args.fetch or not path.exists():
            key = settings().data_go_kr_key
            if not key:
                raise SystemExit(
                    "DATA_GO_KR_KEY 가 없다 — .env 에 공공데이터포털 키(디코딩)를 적는다"
                )
            path.write_text(json.dumps(fetch(op, key), ensure_ascii=False), encoding="utf-8")
        rows_by_op[op] = json.loads(path.read_text(encoding="utf-8"))
        count = len({row["file_nm"] for row in rows_by_op[op]})
        print(f"{OPS[op]}: 장면 {len(rows_by_op[op]):,}줄 · 영상 {count}편")

    # 링크는 클립으로 쓸 영상만 본다.
    links = raw / "alive.json"
    alive: dict[str, bool] = {}
    if links.exists() and not args.fetch:
        alive = json.loads(links.read_text(encoding="utf-8"))
    files = sorted({row["file_nm"] for op in CLIP_OPS for row in rows_by_op[op]})
    todo = [f for f in files if f not in alive]
    if todo:
        print(f"링크 확인 {len(todo)}편…")
        alive.update(check_links(todo))
        links.write_text(json.dumps(dict(sorted(alive.items())), ensure_ascii=False, indent=0))

    table = build(rows_by_op, alive)
    out = settings().release_dir / OUT
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    videos = {r["video_id"] for r in table}
    by_age = collections.Counter(age for _, age in {(r["video_id"], r["age_group"]) for r in table})
    print(f"→ {out.name}: {len(table)}줄 · 영상(=클립) {len(videos)}편")
    print("  연령대별 " + " · ".join(f"{a} {n}" for a, n in sorted(by_age.items())))


if __name__ == "__main__":
    main()
