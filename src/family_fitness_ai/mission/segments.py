"""영상 구간 — 운동 하나를 보여주는 동안 (docs/05).

라벨링은 운동마다 **시작 시각**만 낸다. 미션에는 끝이 있어야 한다 — 몇 분 하는
운동인지 모르면 미션 문구도 완료 판정도 만들 수 없다.

**길이는 시작 시각 사이로 잰다.** 이름표가 사라진 시점으로 재면 운동이 이어지는데도
짧게 나온다 — 이름표는 앞에 잠깐 떴다 사라지는 자막이다.

    end_sec   = 다음 운동의 start_sec        # 마지막 구간은 이름표 끝으로 닫는다
    duration  = end_sec - start_sec
    gap_sec   = end_sec - label_end_sec      # 이름표가 사라진 뒤 남은 빈 구간

`space`·`noise`·`equipment`·`intensity` 는 **넣지 않는다** — 잰 값이 없다.

실행 (labeling.label · rag.chunks 뒤, 저장소 루트에서):
    python -m family_fitness_ai.mission.segments
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..labeling.label import EXERCISES_FILE, LABELING_FILE, watch_url
from ..rag.chunks import CHUNKS_FILE
from ..rag.prescription import identity

SEGMENTS_FILE = "video_segments.csv"  # data/release — 백엔드·검수가 읽는다
STATS_FILE = "segment_stats.csv"  # data/interim — 구간 통계

SEGMENT_COLUMNS = [
    "video_id",
    "start_sec",
    "end_sec",
    "label_end_sec",
    "gap_sec",
    "duration_sec",
    # 길이를 무엇으로 쟀는가. 셋 중 하나다 (LENGTH_BASES)
    "length_basis",
    "last",
    "exercise_name",
    "age_group",
    "source",
    "common",
    "evidence_text",
    "chunk_id",
    "citation_label",
    # 키가 아니라 딸린 값이다 — 계약의 키는 video_id·start_sec 이다 (docs/03 §4.2)
    "url",
]

# `next_start` 다음 운동의 시작으로 쟀다 · `label_end` 마지막 구간이라 이름표 끝으로 닫았다
# · `over_cap` 다음 시작이 영상 길이를 넘어 영상 끝으로 닫았다
LENGTH_BASES = ("next_start", "label_end", "over_cap")

# 근거 원문 앞에 붙은 `03:08` (labeling.label 이 붙인다)
_TIME_MARK = re.compile(r"^\s*\d+:\d{2}\s*")


@dataclass(frozen=True)
class VideoSegment:
    video_id: str
    start_sec: int
    end_sec: int
    label_end_sec: int
    length_basis: str
    last: bool
    exercise_name: str
    age_group: str
    source: str
    common: bool
    evidence_text: str
    chunk_id: str
    citation_label: str

    @property
    def duration_sec(self) -> int:
        return self.end_sec - self.start_sec

    @property
    def gap_sec(self) -> int:
        """이름표가 사라진 뒤 남은 빈 구간. **마지막 구간은 언제나 0 이다** — 그 구간을
        이름표 끝으로 닫았기 때문이다. gap 으로 거르는 검사는 마지막 구간을 하나도
        걸러내지 못한다 (docs/05 §3.2 · `stats` 가 숫자로 낸다)."""
        return self.end_sec - self.label_end_sec

    @property
    def url(self) -> str:
        return watch_url(self.video_id, self.start_sec)


def _exact(name: str, evidence: str) -> bool:
    """이름표 본문이 어휘 이름과 표기 차이까지만 같은가 (부분 일치가 아닌가)."""
    return identity(_TIME_MARK.sub("", evidence)) == identity(name)


def _position(name: str, evidence: str) -> int:
    """이름이 근거 원문에서 나온 자리. 없으면 뒤로 보낸다."""
    text = identity(evidence)
    found = text.find(identity(name))
    return found if found >= 0 else len(text)


def pick_one(rows: list[dict[str, object]]) -> dict[str, object]:
    """같은 `(video_id, start_sec)` 에 이름이 둘일 때 하나만 남긴다.

    **비공통 → 어휘 완전일치 → 근거 원문에 나온 순서** 로 고른다. 실측에서 걸리는 것은
    `척추 들어올리기 (고양이자세)` 꼴 5쌍이다 — 이름 막대의 괄호 속 딴이름이 어휘에도
    있어 두 행이 된다 (docs/02). 둘 다 공통이고 둘 다 통째로는 어휘와 다르니
    앞에 적힌 이름(`척추 들어올리기`)이 남는다.
    """
    return min(
        rows,
        key=lambda r: (
            bool(r["common"]),
            not _exact(str(r["exercise_name"]), str(r["evidence_text"])),
            _position(str(r["exercise_name"]), str(r["evidence_text"])),
        ),
    )


def build(
    exercises: pd.DataFrame, labeling: pd.DataFrame, chunks: pd.DataFrame
) -> tuple[list[VideoSegment], dict[str, int]]:
    """운동 × 영상 표를 구간으로. 뺀 행은 세어서 보고한다 — 조용히 거르지 않는다."""
    lengths = {
        str(r["video_id"]): int(r["duration_sec"])
        for r in labeling.to_dict("records")
        if str(r["duration_sec"])
    }
    cited = {
        str(r["chunk_id"]).removeprefix("video:"): str(r["citation_label"])
        for r in chunks.to_dict("records")
        if str(r["source"]) == "video"
    }

    dropped = {"시각 없음": 0, "청크 없음": 0, "이름이 둘": 0}
    by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in exercises.to_dict("records"):
        if not str(row["start_sec"]):
            dropped["시각 없음"] += 1
            continue
        if str(row["video_id"]) not in cited:
            # 인용할 청크가 없으면 미션에 쓸 수 없다 (docs/04 §2.2)
            dropped["청크 없음"] += 1
            continue
        key = (str(row["video_id"]), int(row["start_sec"]))
        by_key.setdefault(key, []).append({**row, "common": str(row["common"]) == "True"})

    out: list[VideoSegment] = []
    for video_id in sorted({v for v, _ in by_key}):
        starts = sorted(s for v, s in by_key if v == video_id)
        for i, start in enumerate(starts):
            rows = by_key[(video_id, start)]
            dropped["이름이 둘"] += len(rows) - 1
            row = pick_one(rows)
            label_end = int(str(row["end_sec"]))
            last = i + 1 == len(starts)
            end, basis = (label_end, "label_end") if last else (starts[i + 1], "next_start")
            cap = lengths.get(video_id)
            if cap is not None and end > cap:
                end, basis = cap, "over_cap"
            out.append(
                VideoSegment(
                    video_id=video_id,
                    start_sec=start,
                    end_sec=end,
                    label_end_sec=label_end,
                    length_basis=basis,
                    last=last,
                    exercise_name=str(row["exercise_name"]),
                    age_group=str(row["age_group"]),
                    source=str(row["source"]),
                    common=bool(row["common"]),
                    evidence_text=str(row["evidence_text"]),
                    chunk_id=f"video:{video_id}",
                    citation_label=cited[video_id],
                )
            )
    return out, dropped


def segments_frame(segments: list[VideoSegment]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": s.video_id,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "label_end_sec": s.label_end_sec,
                "gap_sec": s.gap_sec,
                "duration_sec": s.duration_sec,
                "length_basis": s.length_basis,
                "last": s.last,
                "exercise_name": s.exercise_name,
                "age_group": s.age_group,
                "source": s.source,
                "common": s.common,
                "evidence_text": s.evidence_text,
                "chunk_id": s.chunk_id,
                "citation_label": s.citation_label,
                "url": s.url,
            }
            for s in segments
        ],
        columns=SEGMENT_COLUMNS,
    )


def read_segments(path: Path) -> list[VideoSegment]:
    """낸 파일을 다시 읽는다. **없거나 비어 있으면 빈 목록이다** — 미션은 영상이 없어도
    성립한다 (docs/05). 예외를 던지면 영상 없는 미션까지 못 만든다."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if frame.empty:
        return []
    return [
        VideoSegment(
            video_id=str(r["video_id"]),
            start_sec=int(r["start_sec"]),
            end_sec=int(r["end_sec"]),
            label_end_sec=int(r["label_end_sec"]),
            length_basis=str(r["length_basis"]),
            last=str(r["last"]) == "True",
            exercise_name=str(r["exercise_name"]),
            age_group=str(r["age_group"]),
            source=str(r["source"]),
            common=str(r["common"]) == "True",
            evidence_text=str(r["evidence_text"]),
            chunk_id=str(r["chunk_id"]),
            citation_label=str(r["citation_label"]),
        )
        for r in frame.to_dict("records")
    ]


# ── 구간 통계 ────────────────────────────────────────────────────────

GAP_MAX = 60  # docs/05 §3.2 가 쓰겠다고 한 gap 상한
LONG = 100  # 이보다 긴 구간은 행으로 뽑아 보여준다

# docs/05 §3.2 가 적어 둔 값. 나란히 찍어 어디가 다른지 보인다.
CLAIMED = {
    ("전체", "구간 수"): "122",
    ("전체", "gap 중앙"): "6",
    ("전체", "gap ≤ 60 통과"): "94",
    ("gap ≤ 60", "길이 중앙"): "74",
    ("gap ≤ 60", "길이 p25"): "54",
    ("gap ≤ 60", "길이 p75"): "78",
    ("gap ≤ 60", "삼등분 경계"): "66 / 76",
    ("gap ≤ 60", "삼등분 몫"): "32 / 30 / 38%",
}


def _terciles(lengths: list[int]) -> tuple[int, int]:
    series = pd.Series(lengths)
    return int(series.quantile(1 / 3)), int(series.quantile(2 / 3))


def _shares(lengths: list[int], low: int, high: int) -> str:
    n = len(lengths)
    # 경계는 반열림이다 — 길이가 2초 단위로 뭉쳐 있어(프레임을 2초마다 읽었다) 경계값에
    # 12~13개씩 걸린다. 닫힘/열림을 어디에 두는지가 몫을 45/26/30 과 32/30/38 로 가른다.
    counts = [
        sum(1 for x in lengths if x < low),
        sum(1 for x in lengths if low <= x < high),
        sum(1 for x in lengths if high <= x),
    ]
    return " / ".join(f"{c}({round(100 * c / n)}%)" for c in counts) if n else ""


def _num(value: float) -> str:
    """`54.5` 는 그대로, `78.0` 은 `78` 로. 반올림해 놓고 잰 값처럼 쓰지 않는다."""
    return f"{value:g}"


def _spread(group: str, segments: list[VideoSegment]) -> list[tuple[str, str, str]]:
    lengths = [s.duration_sec for s in segments]
    if not lengths:
        return [(group, "구간 수", "0")]
    series = pd.Series(lengths)
    low, high = _terciles(lengths)
    return [
        (group, "구간 수", str(len(lengths))),
        (group, "길이 중앙", _num(series.median())),
        (group, "길이 p25", _num(series.quantile(0.25))),
        (group, "길이 p75", _num(series.quantile(0.75))),
        (group, "삼등분 경계", f"{low} / {high}"),
        (group, "삼등분 몫", _shares(lengths, low, high)),
    ]


def stats(segments: list[VideoSegment]) -> list[tuple[str, str, str]]:
    """`(묶음, 잰 것, 값)`. 문서가 주장한 값은 `stats_frame`·`stats_lines` 가 붙인다."""
    gaps = [s.gap_sec for s in segments]
    passed = [s for s in segments if s.gap_sec <= GAP_MAX]
    lasts = [s for s in segments if s.last]
    rows: list[tuple[str, str, str]] = [
        *_spread("전체", segments),
        ("전체", "gap 중앙", _num(pd.Series(gaps).median()) if gaps else ""),
        ("전체", "gap ≤ 60 통과", str(len(passed))),
        *[
            (
                "전체",
                f"length_basis {basis}",
                str(sum(1 for s in segments if s.length_basis == basis)),
            )
            for basis in LENGTH_BASES
        ],
    ]
    rows += _spread("gap ≤ 60", passed)
    # 결함: 마지막 구간은 end 를 이름표 끝으로 닫으므로 gap 이 언제나 0 이다 →
    # gap 필터가 마지막 구간을 하나도 걸러내지 못한다.
    rows += [
        ("마지막 구간", "구간 수", str(len(lasts))),
        ("마지막 구간", "gap 이 0", str(sum(1 for s in lasts if s.gap_sec == 0))),
        ("마지막 구간", "gap ≤ 60 통과", str(sum(1 for s in lasts if s.gap_sec <= GAP_MAX))),
        (
            "마지막 구간",
            "gap 0 인 구간 중 마지막이 아닌 것",
            str(sum(1 for s in segments if s.gap_sec == 0 and not s.last)),
        ),
        (
            "마지막 구간",
            "길이 p25",
            _num(pd.Series([s.duration_sec for s in lasts]).quantile(0.25)) if lasts else "",
        ),
    ]
    rows += _spread("마지막 구간 뺀 gap ≤ 60", [s for s in passed if not s.last])
    rows += [
        (
            f"길이 {LONG}초 초과",
            f"{s.video_id}@{s.start_sec} {s.exercise_name}",
            f"{s.duration_sec}초 (gap {s.gap_sec} · {s.length_basis})",
        )
        for s in sorted(segments, key=lambda s: -s.duration_sec)
        if s.duration_sec > LONG
    ]
    return rows


def stats_frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"group": g, "metric": m, "measured": v, "claimed": CLAIMED.get((g, m), "")}
            for g, m, v in rows
        ],
        columns=["group", "metric", "measured", "claimed"],
    )


def stats_lines(rows: list[tuple[str, str, str]]) -> list[str]:
    lines: list[str] = []
    group = ""
    for g, metric, value in rows:
        if g != group:
            group = g
            lines.append(f"[{g}]")
        claimed = CLAIMED.get((g, metric), "")
        lines.append(f"  {metric:<30} {value:<28}" + (f"문서 {claimed}" if claimed else ""))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="영상 구간을 내고 gap 필터를 숫자로 잰다")
    ap.add_argument("--interim", default="data/interim", help="라벨·청크가 있고 통계를 낼 곳")
    ap.add_argument("--release", default="data/release", help="구간 표를 낼 곳")
    args = ap.parse_args(argv)

    interim, release = Path(args.interim), Path(args.release)
    read = {"encoding": "utf-8-sig", "dtype": str, "keep_default_na": False}
    segments, dropped = build(
        pd.read_csv(interim / EXERCISES_FILE, **read),
        pd.read_csv(interim / LABELING_FILE, **read),
        pd.read_csv(release / CHUNKS_FILE, **read),
    )

    csv = {"index": False, "encoding": "utf-8-sig"}
    segments_frame(segments).to_csv(release / SEGMENTS_FILE, **csv)
    rows = stats(segments)
    stats_frame(rows).to_csv(interim / STATS_FILE, **csv)

    print(f"구간 {len(segments)} · 영상 {len({s.video_id for s in segments})}")
    print("뺀 행 " + " · ".join(f"{k} {n}" for k, n in dropped.items()))
    print("\n".join(stats_lines(rows)))
    print(f"→ {release / SEGMENTS_FILE} · {interim / STATS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
