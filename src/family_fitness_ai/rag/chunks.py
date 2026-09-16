"""코퍼스 청크를 한 파일로 모은다 (docs/04 · docs/04 §1·§2).

세 소스를 같은 열로 낸다.

- `prescription` — `rag.prescription` 이 만든 처방 청크를 그대로 싣는다
- `video` — `labeling` 이 낸 영상 라벨. **영상 1편이 청크 1개**다
- `criteria` — 등급 기준표를 항목×연령구간×성별로 묶는다

**`chunk_id` 는 내용에서만 나온다** (docs/04 §1). 순번도 수집 시각도 쓰지 않는다 —
재색인 때 같은 원문이 같은 id 를 가져야 저장된 인용이 끊기지 않는다.

**연령이 빈 `prescription`·`video` 청크는 만들지 않는다** (docs/04 §2.2). 이 검사가
`age_group IS NULL` 을 "연령 무관"으로 쓰는 것을 안전하게 만드는 짝이다. 빠뜨린 수는
세어서 보고한다 — 조용히 거르면 코퍼스가 왜 작은지 알 수 없다 (docs/04).

실행 (labeling.label 뒤, 저장소 루트에서):
    python -m family_fitness_ai.rag.chunks
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..stats.items import ITEMS

# **data/release 에 둔다.** 서비스가 런타임에 읽는 파일이고 (색인을 여기서 굽는다),
# `data/interim` 은 `.gitignore` 의 `data/*` 에 걸려 커밋되지 않는다 — 새 클론에서
# 색인을 다시 세울 수 없으면 `coach/messages` 를 배포할 수 없다 (docs/02 §4).
CHUNKS_FILE = "chunks.csv"
# 처방·라벨링이 내는 파일
PRESCRIPTION_FILE = "prescription_chunks.csv"
VIDEO_LABELING_FILE = "video_labeling.csv"
VIDEO_EXERCISES_FILE = "video_exercises.csv"
VIDEOS_FILE = "videos.csv"
# 커밋해 둔 등급 기준표
THRESHOLDS_FILE = "grade_thresholds.csv"

CHUNK_COLUMNS = [
    "chunk_id",
    "source",
    "text",
    "citation_label",
    "citation_url",
    "age_group",
    "fitness_factors",
    "grade",
]

Row = dict[str, Any]


def read_csv(path: Path) -> pd.DataFrame:
    """검수하는 사람이 엑셀로 열 수 있게 낸 파일이다 (docs/02 §4)."""
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def mmss(seconds: str) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def prescription_chunks(chunks: pd.DataFrame) -> tuple[list[Row], int]:
    """처방 청크를 공통 열로 옮긴다. 본문·인용·id 를 그대로 쓴다."""
    out, skipped = [], 0
    for row in chunks.to_dict("records"):
        if not row["age_group"]:
            skipped += 1
            continue
        out.append(
            {
                "chunk_id": row["chunk_id"],
                "source": "prescription",
                "text": row["text"],
                "citation_label": row["citation_label"],
                "citation_url": "",
                "age_group": row["age_group"],
                # 처방문에 요인이 없다
                "fitness_factors": "",
                "grade": "",
            }
        )
    return out, skipped


def video_chunks(
    labels: pd.DataFrame, exercises: pd.DataFrame, videos: pd.DataFrame | None = None
) -> tuple[list[Row], int]:
    """영상 1편 = 청크 1개. 본문에 운동 이름과 그 시각을 싣는다.

    공통 준비·마무리는 뒤에 따로 적는다 — 본운동과 섞이면 어느 영상이나
    같은 문장이 되어 검색이 변별하지 못한다.

    운동 이름이 하나도 안 붙은 영상은 본문이 짧다. **부모 문맥을 붙인다** — 재생목록
    이름과 설명문 첫 줄이다. 50토큰 미만 청크는 검색 잡음이다 (docs/04 §2.1).
    """
    described = (
        {str(r["video_id"]): str(r["description"]) for r in videos.to_dict("records")}
        if videos is not None
        else {}
    )
    by_video: dict[str, list[Any]] = defaultdict(list)
    for row in exercises.itertuples():
        by_video[str(row.video_id)].append(row)

    out, skipped = [], 0
    for row in labels.to_dict("records"):
        video_id, title = str(row["video_id"]), str(row["title"])
        if not row["age_group"]:
            skipped += 1
            continue
        found = by_video.get(video_id, [])
        main = [e for e in found if e.common != "True"]
        common = [e for e in found if e.common == "True"]
        parts = [f"국민체력100 운동영상 · {title}", f"연령대 {row['age_group']}"]
        if row["fitness_factors"]:
            parts.append("체력요인 " + "·".join(row["fitness_factors"].split(";")))
        if row["duration_sec"]:
            parts.append(f"길이 {round(int(row['duration_sec']) / 60)}분")
        if main:
            parts.append("나오는 운동: " + ", ".join(_named(e) for e in main))
        if common:
            parts.append("준비·마무리: " + ", ".join(_named(e) for e in common))
        if not main:
            # 부모 문맥 — 재생목록 이름과 설명문 첫 줄 (docs/04 §2.1)
            if row["playlist_titles"]:
                parts.append(row["playlist_titles"].replace(";", " · "))
            parts += _first_line(described.get(video_id, ""))
        out.append(
            {
                "chunk_id": f"video:{video_id}",
                "source": "video",
                "text": " · ".join(parts),
                "citation_label": f"국민체력100 운동영상 · {title}",
                "citation_url": f"https://www.youtube.com/watch?v={video_id}",
                "age_group": row["age_group"],
                "fitness_factors": row["fitness_factors"],
                "grade": "",
            }
        )
    return out, skipped


def _first_line(description: str) -> list[str]:
    """설명문의 첫 줄. 해시태그 줄은 영상마다 같은 상용구라 건너뛴다."""
    for line in description.splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            return [text[:200]]
    return []


def _named(exercise: Any) -> str:
    """`이름(12:34)`. 시각이 없으면 이름만."""
    name = str(exercise.exercise_name)
    return f"{name}({mmss(exercise.start_sec)})" if exercise.start_sec else name


def criteria_chunks(thresholds: pd.DataFrame) -> list[Row]:
    """항목×연령구간×성별 하나가 청크 하나다 (docs/04 §1).

    **연령 무관이다** (docs/04 §2.2) — 청크의 연령 값은 "누구에게 보여도 되는가"이지
    "누구에 관한 자료인가"가 아니다. 등급 기준표는 누구에게 보여도 된다.
    """
    out: list[Row] = []
    grouped = thresholds.groupby(["item_code", "age_group", "sex"], sort=True)
    for (item_code, age_group, sex), rows in grouped:
        item = ITEMS.get(str(item_code))
        name = str(rows.iloc[0]["item_name"])
        unit = str(rows.iloc[0]["unit"])
        who = f"{age_group} {'여자' if sex == 'F' else '남자'}"
        bands = []
        for (lo, hi, age_unit), band in rows.groupby(["age_lo", "age_hi", "age_unit"], sort=False):
            grades = ", ".join(
                f"{r['grade']}등급 {r['threshold']}{unit}"
                for r in band.sort_values("grade").to_dict("records")
            )
            bands.append(f"{lo}~{hi}{age_unit} {grades}")
        out.append(
            {
                "chunk_id": f"criteria:{item_code}-{age_group}-{sex}",
                "source": "criteria",
                "text": f"국민체력100 인증 기준 · {name}({unit}) · {who}: " + " / ".join(bands),
                "citation_label": f"국민체력100 인증 기준 · {name}",
                "citation_url": "",
                # 연령 무관 — 모든 연령대 검색에서 살아남는다 (docs/04 §2.2)
                "age_group": "",
                "fitness_factors": item.factor if item else "",
                # 한 청크가 1~3등급을 모두 싣는다
                "grade": "",
            }
        )
    return out


def build(interim: Path, release: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    prescription, no_age_prescription = prescription_chunks(read_csv(interim / PRESCRIPTION_FILE))
    video, no_age_video = video_chunks(
        read_csv(interim / VIDEO_LABELING_FILE),
        read_csv(interim / VIDEO_EXERCISES_FILE),
        read_csv(interim / VIDEOS_FILE),
    )
    criteria = criteria_chunks(read_csv(release / THRESHOLDS_FILE))
    frame = pd.DataFrame(prescription + video + criteria, columns=CHUNK_COLUMNS)
    skipped = {"prescription": no_age_prescription, "video": no_age_video}
    return frame.sort_values("chunk_id", ignore_index=True), skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="코퍼스 청크를 한 파일로 모은다")
    ap.add_argument("--interim", default="data/interim", help="재료 청크가 있는 곳")
    ap.add_argument("--release", default="data/release", help="기준표가 있고 청크를 낼 곳")
    args = ap.parse_args(argv)

    interim, release = Path(args.interim), Path(args.release)
    frame, skipped = build(interim, release)
    # utf-8-sig — 검수하는 사람이 엑셀로 연다 (docs/02 §4)
    frame.to_csv(release / CHUNKS_FILE, index=False, encoding="utf-8-sig")

    lengths = frame["text"].str.len()
    print(f"청크 {len(frame):,} → {release / CHUNKS_FILE}")
    for source, count in frame["source"].value_counts().sort_index().items():
        print(f"  {source} {count}")
    print(f"본문 길이 중앙 {int(lengths.median())}자 · 최대 {lengths.max()}자")
    for source, count in skipped.items():
        if count:
            print(f"[알림] 연령이 비어 만들지 않은 {source} 청크 {count}개 (docs/04 §2.2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
