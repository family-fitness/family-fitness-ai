"""화면 한 칸을 다시 읽는다.

    python -m family_fitness_ai.video.ocr IhShIA-WJNE lVd366kW7KI …

처음 읽을 때 정한 두 칸(제목·자막)에 운동 이름이 안 걸리는 영상이 있다. 성인
「4주 프로그램」은 이름표가 화면 왼쪽 아래 흰 상자에 뜨는데, 자막 칸이 x 28%
부터라 딱 그 자리를 비껴갔다. 그런 영상만 이름표 자리를 따로 읽어
data/raw/youtube/screen_name.jsonl 에 덧붙인다. 원래의 screen.jsonl 은 건드리지
않는다.

프레임은 이미 받아 둔 것(data/raw/frames)을 쓴다. 유튜브에 다시 가지 않는다.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from family_fitness_ai.common.settings import settings

SIDECAR = "youtube/screen_name.jsonl"

#: 이름표 자리 (x0, y0, x1, y1) — 화면 비율. 1280×720 에서 왼쪽 아래 흰 상자.
NAME_BOX = (0.03, 0.78, 0.24, 0.92)
INTERVAL_SEC = 2


def read_box(image_path: Path, box: tuple[float, float, float, float]) -> list[str]:
    from ocrmac import ocrmac
    from PIL import Image

    image = Image.open(image_path)
    width, height = image.size
    crop = image.crop(
        (int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height))
    )
    found = ocrmac.OCR(crop, language_preference=["ko-KR", "en-US"]).recognize()
    return [text for text, confidence, _ in found if confidence > 0.3]


def read_video(frames_dir: Path, box: tuple[float, float, float, float]) -> list[dict]:
    frames = []
    for index, path in enumerate(sorted(frames_dir.glob("*.jpg"))):
        frames.append({"t": index * INTERVAL_SEC, "name": read_box(path, box)})
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", help="다시 읽을 영상 id")
    args = parser.parse_args()

    raw = settings().raw_dir
    out = raw / SIDECAR
    kept = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            kept[row["key"]] = row

    for video_id in args.videos:
        frames = read_video(raw / "frames" / video_id, NAME_BOX)
        named = sum(1 for frame in frames if frame["name"])
        kept[video_id] = {
            "key": video_id,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "value": {"interval_sec": INTERVAL_SEC, "box": list(NAME_BOX), "frames": frames},
        }
        print(f"  {video_id}: {len(frames)}장 · 글자 있는 칸 {named}")

    with out.open("w", encoding="utf-8") as fh:
        for row in kept.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
