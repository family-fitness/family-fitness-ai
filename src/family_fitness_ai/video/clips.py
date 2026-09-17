"""영상 → 클립.

    python -m family_fitness_ai.video.clips

한 영상에 운동이 여럿 들어 있다. 화면에 뜨는 운동 이름이 바뀌는 지점마다 끊어
data/release/video_clips.csv 를 만든다. 유튜브에 다시 가지 않는다 — 이미 읽어 둔
화면 글자(data/raw/youtube/screen.jsonl)만 쓴다.

화면에는 세 가지 글자가 같이 뜬다.

    배너    「비대면 표준 운동 프로그램」 — 영상 내내 떠 있다
    단계    「준비운동」 — 한 토막 동안 떠 있다
    이름    「상체숙여 발목 잡기」 — 한 운동 동안 떠 있다
    해설    「한 쪽 다리당 20초씩 실시해줍니다」 — 두어 칸 만에 바뀐다

셋을 가르는 것은 **얼마나 오래 떠 있느냐**다. 영상 내내면 배너, 두어 칸이면
해설, 그 사이가 이름이다. OCR 이 흔들려 같은 글자가 조금씩 다르게 읽히므로
비슷한 글자는 하나로 모은다.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from family_fitness_ai.common.settings import ROOT, settings

SCREEN = "youtube/screen.jsonl"

#: 워터마크와 OCR 부스러기. 「국민체력100」이 프레임마다 다르게 읽힌다.
_JUNK = re.compile(
    r"(국민|굿편|지단|국코|코민|굳민|국먼)\s*.{0,3}\s*(체력|에릭|예약|의력|해리|에서|릭|력)"
    r"|^\W*\d{0,3}\W*$"
    r"|new\s*bol?a?nce"
    r"|^[A-Za-z0-9\W]{1,5}$"
)
_PHASE = re.compile(r"(준비\s*운동|본\s*운동|정리\s*운동|마무리\s*운동|쿨\s*다운)")

#: 영상 내내 떠 있으면 배너다.
BANNER_SHARE = 0.85
#: 이만큼 **연달아** 못 버티면 해설이다 (2초 간격이니 6칸 = 12초). 해설도 영상
#: 내내 여러 번 나올 수 있어, 몇 번 나왔나가 아니라 한 번에 얼마나 붙어 있었나로
#: 가른다.
NAME_MIN_RUN = 6
#: 운동 이름이 이보다 길면 이름이 아니라 문장이다.
NAME_MAX_CHARS = 28
#: 이보다 짧은 토막은 클립으로 세지 않는다.
CLIP_MIN_SEC = 20
#: 한 클립이 이보다 길면 이름이 안 바뀐 것이지 한 동작이 아니다.
CLIP_MAX_SEC = 240
#: 이름이 잠깐 사라졌다 돌아오면 같은 클립으로 잇는다.
GAP_FRAMES = 3


@dataclass(frozen=True)
class Clip:
    video_id: str
    seq: int
    name: str
    phase: str
    start_sec: int
    end_sec: int

    @property
    def duration_sec(self) -> int:
        return self.end_sec - self.start_sec


#: 「01. 」 「#」 같은 앞머리. 화면에는 뜨지만 운동 이름은 아니다.
_LEAD = re.compile(r"^\s*(?:#|\d{1,2}\s*[.)]\s*)+")


def _tidy(text: str) -> str:
    text = _LEAD.sub("", text)
    text = text.strip(" |·•ㅣ")
    text = text.replace("〈", "(").replace("〉", ")").replace("<", "(").replace(">", ")")
    text = re.sub(r"\s*([·•])\s*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _pieces(frame: dict, field: str) -> list[str]:
    out = []
    for text in frame.get(field) or []:
        for piece in str(text).split("/"):
            piece = _tidy(piece)
            if piece and not _JUNK.search(piece):
                out.append(piece)
    return out


def _canonical(counts: collections.Counter[str]) -> dict[str, str]:
    """OCR 이 흔들린 글자를 가장 많이 나온 꼴로 모은다."""
    canon: dict[str, str] = {}
    ordered = [text for text, _ in counts.most_common()]
    for text in ordered:
        match = next(
            (
                kept
                for kept in canon.values()
                if difflib.SequenceMatcher(None, text, kept).ratio() >= 0.8
            ),
            None,
        )
        canon[text] = match or text
    return canon


def read_screens(raw_dir: Path) -> dict[str, dict]:
    """영상마다 한 벌씩. 자막 칸까지 읽은 판이 있으면 그쪽을 쓴다."""
    best: dict[str, dict] = {}
    with (raw_dir / SCREEN).open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            value = row["value"]
            if isinstance(value, str):
                value = ast.literal_eval(value)
            if "boxes" in value or row["key"] not in best:
                best[row["key"]] = value
    return best


def _runs_of(sequence: list[str]) -> collections.Counter[str]:
    """이름마다 한 번에 몇 칸을 붙어 있었는지 가장 긴 것."""
    best: collections.Counter[str] = collections.Counter()
    run = 0
    for index, name in enumerate(sequence):
        if index and name == sequence[index - 1]:
            run += 1
        else:
            run = 1
        if name:
            best[name] = max(best[name], run)
    return best


def _bottom_line(pieces: list[str], canon: dict[str, str]) -> str:
    """화면 한 칸에서 운동 이름 자리. 늘 맨 아랫줄이다.

    제목 칸은 「프로그램 이름 / 단계 / 운동 이름」 순으로 쌓이고, 자막 칸은
    「해설 / 운동 이름」 순으로 쌓인다. OCR 이 읽는 차례가 곧 화면의 위아래라
    맨 뒤가 이름이다.

    맨 아랫줄이 단계이거나 문장이면 이 칸에는 이름이 없는 것이다. 한 줄 위로
    거슬러 올라가지 않는다 — 올라가면 프로그램 이름을 운동 이름으로 읽는다.
    """
    for piece in reversed(pieces):
        text = canon[piece]
        if _PHASE.search(text) or len(text) > NAME_MAX_CHARS:
            return ""
        return text
    return ""


def _filled(names: list[str]) -> list[str]:
    """이름이 잠깐 끊겨도 같은 이름이 곧 돌아오면 이어 붙인다."""
    out = list(names)
    for i, name in enumerate(out):
        if name:
            continue
        back = next((out[j] for j in range(i - 1, max(-1, i - 1 - GAP_FRAMES), -1) if out[j]), "")
        ahead = next(
            (names[j] for j in range(i + 1, min(len(names), i + 1 + GAP_FRAMES)) if names[j]),
            "",
        )
        if back and back == ahead:
            out[i] = back
    return out


def _phases(seen: list[tuple[list[str], list[str]]]) -> list[str]:
    """화면에 뜬 단계. 준비와 정리 사이에 아무 말이 없으면 본운동이다."""
    marks = []
    for titles, bars in seen:
        found = next((_PHASE.search(t) for t in titles + bars if _PHASE.search(t)), None)
        marks.append(re.sub(r"\s+", "", found.group(0)) if found else "")
    out = list(marks)
    last = ""
    for i, mark in enumerate(out):
        if mark:
            last = mark
        elif last:
            out[i] = last
    # 준비운동이 끝나고 정리운동이 뜨기 전까지는 본운동이다.
    later = ""
    for i in range(len(out) - 1, -1, -1):
        if marks[i]:
            later = marks[i]
        elif out[i] == "준비운동" and later in ("정리운동", "마무리운동"):
            out[i] = "본운동"
    return out


def clips_of_video(video_id: str, screen: dict) -> list[Clip]:
    frames = screen["frames"]
    if not frames:
        return []
    step = int(screen.get("interval_sec", 2))
    total = len(frames)

    seen = [(_pieces(f, "texts"), _pieces(f, "bar")) for f in frames]
    counts = collections.Counter(t for titles, bars in seen for t in set(titles + bars))
    canon = _canonical(counts)

    sources = {}
    for field, index in (("texts", 0), ("bar", 1)):
        picked = [_bottom_line(pair[index], canon) for pair in seen]
        runs = _runs_of(picked)
        names = {name for name in picked if name and runs[name] >= NAME_MIN_RUN}
        # 영상 내내 떠 있는 것은 배너다 — 다만 이름이 그것 하나뿐이면 배너가
        # 아니라 운동이 하나뿐인 영상이다. 그 하나를 지우면 남는 게 없다.
        if len(names) > 1:
            shares = collections.Counter(name for name in picked if name)
            names -= {n for n in names if shares[n] >= BANNER_SHARE * total}
        sources[field] = _filled([name if name in names else "" for name in picked])

    titles, bars = sources["texts"], sources["bar"]
    distinct_title = len({n for n in titles if n})
    distinct_bar = len({n for n in bars if n})
    # 제목 칸이 프로그램 이름만 물고 있는 영상이 있다. 그럴 때는 자막 칸의 보라색
    # 이름표가 유일한 단서다 — 이름이 훨씬 많이 나오는 쪽을 믿는다.
    picked = bars if distinct_bar >= max(3, 2 * distinct_title) else titles
    phases = _phases(seen)

    spans: list[list[Any]] = []
    for index, name in enumerate(picked):
        second = frames[index].get("t", index * step)
        if spans and spans[-1][0] == name:
            spans[-1][2] = second + step
        else:
            spans.append([name, second, second + step, phases[index]])

    found: list[Clip] = []
    for name, start, end, phase in spans:
        if not name or end - start < CLIP_MIN_SEC:
            continue
        end = int(min(end, start + CLIP_MAX_SEC))
        found.append(Clip(video_id, len(found) + 1, name, phase, int(start), end))
    return found


def build(raw_dir: Path, out_dir: Path) -> list[Clip]:
    screens = read_screens(raw_dir)
    clips: list[Clip] = []
    for video_id, screen in sorted(screens.items()):
        clips.extend(clips_of_video(video_id, screen))

    path = out_dir / "video_clips.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "video_id",
                "seq",
                "name_on_video",
                "phase_on_video",
                "start_sec",
                "end_sec",
                "duration_sec",
            ]
        )
        for clip in clips:
            writer.writerow(
                [
                    clip.video_id,
                    clip.seq,
                    clip.name,
                    clip.phase,
                    clip.start_sec,
                    clip.end_sec,
                    clip.duration_sec,
                ]
            )
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out-dir", default=str(settings().release_dir))
    args = parser.parse_args()

    clips = build(Path(args.raw_dir), Path(args.out_dir))
    videos = {clip.video_id for clip in clips}
    lengths = sorted(clip.duration_sec for clip in clips)
    middle = lengths[len(lengths) // 2] if lengths else 0
    print(f"클립 {len(clips)}개 · 영상 {len(videos)}편 · 길이 중앙값 {middle}초")


if __name__ == "__main__":
    main()
