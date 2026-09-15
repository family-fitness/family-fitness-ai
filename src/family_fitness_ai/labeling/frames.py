"""영상 화면에서 운동 이름표를 읽는다 (docs/dev/AI-7 §2.1).

공단 영상은 구간마다 운동 이름을 화면에 띄운다. 자리는 두 곳이다.

- 왼쪽 위 이름표 — 윗줄이 체력요인, 아랫줄이 운동 이름 (`(응용편) 민첩성·순발력·협응성`)
- 아래 가운데 이름 막대 — 청소년·유소년 「비대면 표준 운동 프로그램」 (§3.9)

**프레임 추출은 이 모듈에서만 한다** (docs/02 §3.2). **영상 파일은 저장하지 않는다** —
yt-dlp 가 받는 바이트를 파이프로 ffmpeg 에 흘려 프레임만 뽑는다. 프레임은 30일이 지나면
지운다 (docs/01 §4.3). 읽은 글자는 JSONL 에 캐시한다. 읽는 자리가 늘면 디스크에 남은
프레임에서 새 자리만 읽는다 — YouTube 에 다시 가지 않는다.

글자 인식은 macOS Vision(`ocrmac`)이라 **맥에서만 돈다.** yt-dlp·ffmpeg·deno 가 필요하다.
읽은 글자를 구간으로 묶고 라벨로 옮기는 규칙은 `label` 에 있다.

실행 (collect 뒤, 저장소 루트에서 · 맥이 잠들면 네트워크가 끊긴다):
    caffeinate -i python -m family_fitness_ai.labeling.frames                # videos.csv 전부
    caffeinate -i python -m family_fitness_ai.labeling.frames --video <ID>   # 몇 편만
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from . import collect as C
from . import label as L

FRAMES_DIR = Path("data/raw/frames")
FRAME_INTERVAL_SEC = 2
FRAME_RETENTION_DAYS = 30  # docs/01 §4.3
DOWNLOAD_DELAY_SEC = 5.0
VIDEO_FORMAT = "bv*[height<=720][ext=mp4]/b[height<=720]"

# 읽는 자리 — (왼, 위, 오른, 아래) 비율. 캐시에는 이 이름으로 자리마다 글자를 둔다.
BOXES = {
    # 왼쪽 위 이름표. 이 밖은 로고·노래 가사다.
    "texts": (0.0, 0.0, 0.45, 0.25),
    # 아래 가운데 이름 막대. 왼쪽 아래 `자극부위` 그림과 심박 표는 뺀다.
    "bar": (0.28, 0.78, 0.80, 0.97),
}

# 영상 자체가 없다고 yt-dlp 가 말하는 문구. **이때만** 사유를 캐시한다.
# 그 밖의 실패(막힘·네트워크 끊김·알 수 없음)는 캐시하지 않고 멈춘다 — 2026-09-14 네트워크가
# 잠깐 끊겼을 때 20편이 "받을 수 없음"으로 캐시돼 다시 돌려도 건너뛰게 됐다.
GONE_MARKERS = ("Private video", "Video unavailable", "has been removed", "members-only")

# (보내는 쪽 종료 코드, 보내는 쪽 오류, 받는 쪽 종료 코드, 받는 쪽 오류)
PipeResult = tuple[int, str, int, str]
Pipe = Callable[[list[str], list[str]], PipeResult]


class Blocked(RuntimeError):
    """YouTube 에 닿지 못했다 (막힘·네트워크). 캐시하지 않는다 — 다시 돌리면 이어서 읽는다."""


class DownloadFailed(RuntimeError):
    """이 영상은 없다 (비공개·삭제). 사유를 캐시한다."""


def _pipe(source: list[str], sink: list[str]) -> PipeResult:
    """`source | sink`. 영상 바이트는 파이프로만 흐르고 디스크에 닿지 않는다.

    보내는 쪽 오류는 임시 파일로 받는다 — 파이프로 받으면 버퍼가 차서 둘 다 멈출 수 있다.
    """
    with tempfile.TemporaryFile() as source_err:
        sender = subprocess.Popen(source, stdout=subprocess.PIPE, stderr=source_err)
        assert sender.stdout is not None
        receiver = subprocess.Popen(
            sink, stdin=sender.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        sender.stdout.close()  # 받는 쪽이 먼저 끝나면 보내는 쪽도 끝나게
        _, sink_err = receiver.communicate()
        sender.wait()
        source_err.seek(0)
        sent_err = source_err.read().decode(errors="replace")
    return sender.returncode, sent_err, receiver.returncode, sink_err


def stream_frames(
    video_id: str,
    out_dir: Path,
    interval_sec: int = FRAME_INTERVAL_SEC,
    pipe: Pipe = _pipe,
) -> list[tuple[int, Path]]:
    """**영상 파일을 쓰지 않고** 프레임만 뽑는다. `(초, 프레임 파일)` 목록."""
    shutil.rmtree(out_dir, ignore_errors=True)  # 끊긴 실행이 남긴 프레임
    out_dir.mkdir(parents=True)
    source = ["yt-dlp", "-q", "--no-warnings", "--no-playlist", "-f", VIDEO_FORMAT, "-o", "-"]
    source.append(f"https://www.youtube.com/watch?v={video_id}")
    sink = ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-vf", f"fps=1/{interval_sec}"]
    sink += ["-q:v", "3", str(out_dir / "%05d.jpg")]

    source_code, source_err, sink_code, sink_err = pipe(source, sink)
    if source_code != 0 or sink_code != 0:
        shutil.rmtree(out_dir, ignore_errors=True)  # 반쪽 프레임으로 캐시하지 않는다
        if source_code != 0 and any(marker in source_err for marker in GONE_MARKERS):
            raise DownloadFailed(_error_line(source_err))
        raise Blocked(_error_line(source_err if source_code != 0 else sink_err))
    return frames_on_disk(out_dir, interval_sec)


def frames_on_disk(out_dir: Path, interval_sec: int = FRAME_INTERVAL_SEC) -> list[tuple[int, Path]]:
    """이미 뽑은 프레임. 프레임 번호에서 시각을 낸다."""
    return [((int(p.stem) - 1) * interval_sec, p) for p in sorted(out_dir.glob("*.jpg"))]


def _error_line(stderr: str) -> str:
    """마지막 오류 한 줄. 스트림 주소에는 서명이 들어 있어 지운다."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("ERROR")]
    line = (errors or lines or ["알 수 없음"])[-1]
    return re.sub(r"https?://\S+", "<주소>", line)[:200]


def needs_reading(cached: Mapping[str, Any] | None) -> bool:
    """캐시에 없거나, 읽은 뒤 늘어난 자리가 비어 있으면 읽는다.

    영상이 없다는 캐시는 다시 보지 않는다.
    """
    if cached is None:
        return True
    if "missing" in cached:
        return False
    frames = cached.get("frames", [])
    return bool(frames) and any(field not in frames[0] for field in BOXES)


def read_overlay(
    frames: list[tuple[int, Path]], known: Mapping[int, Mapping[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """프레임마다 자리별 글자. 자리 안에서는 위에서 아래 순서다.

    `known` 에 이미 읽은 자리가 있으면 그 자리는 다시 읽지 않는다.
    """
    # 맥에서만 있다. 모듈 맨 위에서 부르면 CI(리눅스)가 이 모듈을 불러오지 못한다.
    from ocrmac import ocrmac
    from PIL import Image

    out: list[dict[str, Any]] = []
    for t, path in frames:
        entry: dict[str, Any] = {"t": t, **(known or {}).get(t, {})}
        missing = [field for field in BOXES if field not in entry]
        if missing:
            with Image.open(path) as image:
                w, h = image.size
                for field in missing:
                    left, top, right, bottom = BOXES[field]
                    box = image.crop((int(left * w), int(top * h), int(right * w), int(bottom * h)))
                    found = ocrmac.OCR(
                        box, recognition_level="accurate", language_preference=["ko-KR"]
                    ).recognize()
                    # Vision 의 좌표는 아래가 0 이다. 위에서 아래로 늘어놓는다.
                    found.sort(key=lambda r: (-(r[2][1] + r[2][3]), r[2][0]))
                    entry[field] = [text for text, _, _ in found]
        out.append(entry)
    return out


def purge_old_frames(root: Path, days: int = FRAME_RETENTION_DAYS, now: float | None = None) -> int:
    """만든 지 `days` 일이 지난 프레임 디렉터리를 지운다 (docs/01 §4.3). 지운 편수."""
    if not root.exists():
        return 0
    cutoff = (time.time() if now is None else now) - days * 86400
    old = [d for d in root.iterdir() if d.is_dir() and d.stat().st_mtime < cutoff]
    for d in old:
        shutil.rmtree(d)
    return len(old)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="영상 화면의 운동 이름표를 읽는다 (맥에서만)")
    ap.add_argument("--video", action="append", help="이 영상만 (여러 번). 없으면 videos.csv 전부")
    ap.add_argument("--interim", default="data/interim", help="videos.csv 가 있는 곳")
    ap.add_argument("--cache", default=str(C.CACHE_DIR), help="읽은 글자 캐시 (커밋하지 않는다)")
    ap.add_argument("--frames", default=str(FRAMES_DIR), help="프레임을 둘 곳 (30일 뒤 지운다)")
    args = ap.parse_args(argv)

    frames_root = Path(args.frames)
    purged = purge_old_frames(frames_root)
    videos = pd.read_csv(
        Path(args.interim) / C.VIDEOS_FILE, encoding="utf-8-sig", dtype=str, keep_default_na=False
    )
    wanted = [
        str(row["video_id"])
        for row in videos.to_dict("records")
        if not L.EXCLUDED_PLAYLISTS.intersection(str(row["playlist_ids"]).split(";"))
        and (not args.video or row["video_id"] in args.video)
    ]

    cache = C.JsonlCache(Path(args.cache) / C.SCREEN_CACHE)
    streamed = reread = failed = 0
    for i, vid in enumerate(wanted, 1):
        cached = cache.get(vid)
        if not needs_reading(cached):
            continue
        started = time.time()
        frames = frames_on_disk(frames_root / vid) if cached else []
        if not frames:
            try:
                frames = stream_frames(vid, frames_root / vid)
            except Blocked as e:
                print(f"[중단] YouTube 에 닿지 못했다 ({vid} · {e}). 다시 돌리면 이어서 읽는다")
                break
            except DownloadFailed as e:
                cache.put(vid, {"missing": str(e)})
                failed += 1
                continue
        known = {int(f["t"]): f for f in (cached or {}).get("frames", [])}
        cache.put(
            vid,
            {
                "interval_sec": FRAME_INTERVAL_SEC,
                "boxes": {field: list(box) for field, box in BOXES.items()},
                "frames": read_overlay(frames, known),
            },
        )
        took = time.time() - started
        how = "디스크의 프레임" if known else "스트림"
        print(
            f"  {i}/{len(wanted)} {vid} · {how} · 프레임 {len(frames)} · {took:.0f}초", flush=True
        )
        if known:
            reread += 1
        else:
            streamed += 1
            time.sleep(DOWNLOAD_DELAY_SEC)

    done = sum(1 for vid in wanted if not needs_reading(cache.get(vid)))
    print(
        f"화면 읽음 {done}/{len(wanted)} · 스트림 {streamed} · 디스크 {reread} · 영상 없음 {failed}"
    )
    print(f"30일 지난 프레임 {purged}편을 지웠다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
