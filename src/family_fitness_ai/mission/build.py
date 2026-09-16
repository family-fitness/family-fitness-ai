"""미션 집합 (docs/05).

    미션 = 운동 × 영상 구간 × 라벨

**이 표가 추천의 검색 대상이다.** 요청이 오면 여기서 라벨로 좁혀 미션을 찾는다 —
요청마다 처방 칸에서 새로 조립하지 않는다. 프론트의 `[미션 추천]` 카드가
`운동 이름 + 영상 링크 + 짧은 문구`인데, 앞의 둘이 이 표의 한 행이다.

**값을 복사해 넣지 않는다.** 운동 이름과 구간은 `prescription_cells.csv` ·
`video_segments.csv` 에 있고 여기는 **잇는 자리**다. `mission_id` 가 결정적이라
두 번 돌려도 같은 표가 난다.

**영상이 없는 운동도 미션이다** (docs/03 §5.7 — `video: null` 은 결함이 아니다).
한 운동에 구간이 여럿이면 **구간마다 한 미션**이다 — 같은 운동이라도 영상이 다르면
다른 카드다.

실행 (mission.cells · mission.segments 뒤):
    python -m family_fitness_ai.mission.build
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..common.settings import RELEASE_DIR
from ..labeling.label import watch_url
from ..rag.prescription import identity
from .cells import Cell, chunk_prefix, load_cells
from .segments import SEGMENTS_FILE, VideoSegment, read_segments

MISSIONS_FILE = "missions.csv"

MULTI = ";"  # 다중값 구분자 (docs/02 §4 ②)

MISSION_COLUMNS = [
    "mission_id",
    "exercise_id",
    "exercise_name",
    "video_id",
    "start_sec",
    "end_sec",
    # `video_id` 로 만든 재생 주소. **키가 아니라 딸린 값이다**
    "url",
    # **영상의 연령대.** 연령 필터의 입력이다 (`select.video_allowed` · docs/02 §2.3) —
    # 이 열이 없으면 표에서 읽은 미션의 영상이 「연령 미상」이 되어 아이에게 하나도
    # 나가지 않는다 (`suitableFor` 가 빈 연령을 막는다). 운동의 대상(`targets`)과
    # 다른 값이다 — 유아기 영상이 만 8세에게 나가는 예외가 있다 (§2.3)
    "video_age_group",
    # 대상 라벨. `연령대-성별` 목록 (등급 축은 아직 없다 — 아래)
    "targets",
    "phase",
    "common",
    # 그 대상 칸에서 이 운동이 몇 번째로 많이 처방됐나 (가장 앞선 것)
    "best_rank",
    "prescribed",
    # 근거 — 미션 하나가 어느 청크에서 왔나
    "cell_chunk_id",
    "cell_citation_label",
    "video_chunk_id",
    "video_citation_label",
]

# **채우지 않는 열**과 그 이유. 더미로 메우지 않는다 (AGENTS.md §4).
#
# `intensity` — 구간 길이 삼등분이 근거인데 `gap` 필터가 마지막 구간을 구조적으로
#   못 걸러낸다. 그 결함이 정리되기 전에는 쓰지 않는다
# `indoor` — 운동 표(`exercises.csv`)에서 와야 하는데 ① 어휘 정제가 보류다
# `targets` 의 항목·등급 — `prescription_cells.csv` 에 `item_code`·`grade` 축이
#   없다 — 등급 계산을 아직 붙이지 않았다. 지금은 `연령대-성별` 까지다
NOT_FILLED = ("intensity", "indoor")


@dataclass(frozen=True)
class Mission:
    """미션 한 행. 영상이 없으면 `segment` 가 비어 있다."""

    exercise_id: str
    exercise_name: str
    segment: VideoSegment | None
    # `연령대-성별` 목록. 한 운동이 여러 칸에 처방되므로 여럿이다
    targets: tuple[str, ...]
    phase: str
    best_rank: int
    prescribed: int
    cell_chunk_id: str
    cell_citation_label: str

    @property
    def mission_id(self) -> str:
        """`{exercise_id}@{video_id}#{start_sec}` — 결정적이다.

        영상이 없으면 뒤를 비운다 — 그 운동의 미션이 하나뿐이라는 뜻이다.
        """
        if self.segment is None:
            return f"{self.exercise_id}@"
        return f"{self.exercise_id}@{self.segment.video_id}#{self.segment.start_sec}"

    @property
    def common(self) -> bool:
        """공통 준비·마무리 구간인가 — 순위를 낮춘다. 영상이 없으면 모른다."""
        return self.segment.common if self.segment else False


def target(age_group: str, sex: str) -> str:
    return f"{age_group}-{sex}"


def build(cells: list[Cell], segments: list[VideoSegment]) -> list[Mission]:
    """미션 전부. **칸의 청크에 든 이름만** 쓴다 (`chunk_prefix`) — 잘린 뒤의 이름을
    고르면 인용한 청크가 그 운동을 말하지 않는다."""
    by_name: dict[str, list[VideoSegment]] = defaultdict(list)
    for segment in segments:
        by_name[identity(segment.exercise_name)].append(segment)

    # 운동 하나에 대상·단계·순위를 모은다
    targets: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    best: dict[str, tuple[int, str, str, str]] = {}
    total: dict[str, int] = defaultdict(int)

    for cell in cells:
        keep = chunk_prefix(cell)
        for rank, (name, count) in enumerate(
            zip(cell.exercise_names[:keep], cell.exercise_counts[:keep], strict=True), start=1
        ):
            key = identity(name)
            display.setdefault(key, name)
            targets[key].add(target(cell.age_group, cell.sex))
            total[key] += count
            # 가장 앞선 순위와 그때의 근거를 남긴다 — 추천이 순위로 정렬한다
            if key not in best or rank < best[key][0]:
                best[key] = (rank, cell.phase, cell.chunk_id, cell.citation_label)

    out: list[Mission] = []
    for key, name in sorted(display.items(), key=lambda kv: kv[1]):
        rank, phase, chunk_id, label = best[key]
        common = dict(
            exercise_id=key,
            exercise_name=name,
            targets=tuple(sorted(targets[key])),
            phase=phase,
            best_rank=rank,
            prescribed=total[key],
            cell_chunk_id=chunk_id,
            cell_citation_label=label,
        )
        found = by_name.get(key, [])
        if not found:
            out.append(Mission(segment=None, **common))  # type: ignore[arg-type]
            continue
        # 구간마다 한 미션이다 — 같은 운동이라도 영상이 다르면 다른 카드다
        for segment in sorted(found, key=lambda s: (s.video_id, s.start_sec)):
            out.append(Mission(segment=segment, **common))  # type: ignore[arg-type]
    return out


def missions_frame(missions: list[Mission]) -> pd.DataFrame:
    rows = []
    for mission in missions:
        segment = mission.segment
        rows.append(
            {
                "mission_id": mission.mission_id,
                "exercise_id": mission.exercise_id,
                "exercise_name": mission.exercise_name,
                "video_id": "" if segment is None else segment.video_id,
                "start_sec": None if segment is None else segment.start_sec,
                "end_sec": None if segment is None else segment.end_sec,
                "url": "" if segment is None else watch_url(segment.video_id, segment.start_sec),
                "video_age_group": "" if segment is None else segment.age_group,
                "targets": MULTI.join(mission.targets),
                "phase": mission.phase,
                "common": mission.common,
                "best_rank": mission.best_rank,
                "prescribed": mission.prescribed,
                "cell_chunk_id": mission.cell_chunk_id,
                "cell_citation_label": mission.cell_citation_label,
                "video_chunk_id": "" if segment is None else segment.chunk_id,
                "video_citation_label": "" if segment is None else segment.citation_label,
            }
        )
    frame = pd.DataFrame(rows, columns=MISSION_COLUMNS)
    for column in ("start_sec", "end_sec"):
        frame[column] = frame[column].astype("Int64")  # 96.0 으로 쓰이지 않게
    return frame


def read_missions(path: Path = RELEASE_DIR / MISSIONS_FILE) -> list[Mission]:
    """낸 표를 다시 읽는다. **영상 칸이 비면 `segment` 가 `None` 이다.**

    `VideoSegment` 를 온전히 되살리지는 않는다 — 추천에 필요한 것은 `video_id` ·
    `start_sec` · 인용뿐이고, 나머지는 `video_segments.csv` 가 정본이다.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    out: list[Mission] = []
    for row in frame.to_dict("records"):
        video_id = str(row["video_id"])
        segment = (
            None
            if not video_id
            else VideoSegment(
                video_id=video_id,
                start_sec=int(row["start_sec"]),
                end_sec=int(row["end_sec"]),
                label_end_sec=int(row["end_sec"]),
                length_basis="",
                last=False,
                exercise_name=str(row["exercise_name"]),
                age_group=str(row["video_age_group"]),
                source="",
                common=str(row["common"]) == "True",
                evidence_text="",
                chunk_id=str(row["video_chunk_id"]),
                citation_label=str(row["video_citation_label"]),
            )
        )
        out.append(
            Mission(
                exercise_id=str(row["exercise_id"]),
                exercise_name=str(row["exercise_name"]),
                segment=segment,
                targets=tuple(x for x in str(row["targets"]).split(MULTI) if x),
                phase=str(row["phase"]),
                best_rank=int(row["best_rank"]),
                prescribed=int(row["prescribed"]),
                cell_chunk_id=str(row["cell_chunk_id"]),
                cell_citation_label=str(row["cell_citation_label"]),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="미션 집합을 CSV로 낸다")
    ap.add_argument("--release", default=str(RELEASE_DIR), help="표를 낼 곳 (커밋한다)")
    args = ap.parse_args(argv)

    release = Path(args.release)
    cells = load_cells(release / "prescription_cells.csv")
    segments = read_segments(release / SEGMENTS_FILE)
    missions = build(cells, segments)

    out = release / MISSIONS_FILE
    # utf-8-sig — 검수하는 사람이 엑셀로 연다 (docs/02 §4 ③)
    missions_frame(missions).to_csv(out, index=False, encoding="utf-8-sig")

    with_video = sum(1 for m in missions if m.segment is not None)
    print(
        f"미션 {len(missions)} · 영상 붙은 것 {with_video} · 없는 것 {len(missions) - with_video}"
    )
    print(f"고유 운동 {len({m.exercise_id for m in missions})}")
    print(f"mission_id 가 겹치는 것 {len(missions) - len({m.mission_id for m in missions})}")
    groups: dict[str, int] = defaultdict(int)
    for mission in missions:
        for label in mission.targets:
            groups[label.split("-")[0]] += 1
    print("대상 연령대별 미션 수 " + " · ".join(f"{k} {v}" for k, v in sorted(groups.items())))
    print(f"채우지 않은 열: {', '.join(NOT_FILLED)} (더미로 메우지 않는다)")
    print(f"표 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
