"""클립 라벨과 시작 · 끝 시각을 화면에서 보고 고친다.

    python src/script/label_clips.py            # http://127.0.0.1:8765 를 브라우저로 연다
    python src/script/label_clips.py --port 9000 --no-open

data/release/clip_labels.csv 는 334줄이고, csv 를 한 줄씩 읽으며 영상을 따로 찾아 보는
것이 번거로웠다. 이름을 고르면 그 이름의 유튜브 클립이 그 구간만 재생되고, 옆에서 라벨을
고쳐 저장한다. 저장한 줄은 source 가 human 이 된다 — video.labels 가 다시 돌려도 지키는
줄이다. 고치지 않고 저장해도 human 이 된다. 「사람이 보고 확인했다」는 뜻이다.

클립의 시작 · 끝도 고친다. 고친 시각은 video_clips.csv 에 바로 적고, clip_times.csv 에도
남긴다 — make clips 를 다시 돌리면 video_clips.csv 가 새로 쓰이는데, 그때 video.clips 가
clip_times.csv 의 값을 그 위에 덮어쓴다.

이 PC 에서만 쓴다. 127.0.0.1 에만 열고, LLM 도 유튜브 API 도 부르지 않는다(재생은 유튜브
재생기가 한다). 공단 영상(kspo_videos.csv)은 다루지 않는다 — 그 표는 make kspo 가 API 값으로
다시 만든다.
"""

from __future__ import annotations

import argparse
import csv
import json
import webbrowser
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from family_fitness_ai.common.copy import video_title
from family_fitness_ai.common.items import FACTORS
from family_fitness_ai.common.settings import settings
from family_fitness_ai.rag.index import corpus
from family_fitness_ai.video import clips as clip_table
from family_fitness_ai.video import vocabulary
from family_fitness_ai.video.labels import PHASES

PAGE = Path(__file__).with_name("label_clips.html")

#: 화면에서 고치는 칸. 나머지(name_on_video · source · score)는 도구가 정한다.
TEXT_FIELDS = ("exercise_name", "fitness_factor", "phase")
FLAG_FIELDS = ("is_exercise", "home_ok", "quiet", "needs_props")

#: 사람이 고친 줄의 점수. 표에 이미 있는 human 줄이 모두 이 값이다.
HUMAN_SCORE = "0.0"


class LabelError(ValueError):
    """받을 수 없는 라벨. 화면에 그대로 보여 줄 말을 담는다."""


def read_labels(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or ()), list(reader)


def _checked(changes: Mapping[str, object]) -> dict[str, str]:
    """화면에서 온 값을 표에 적을 글자로 바꾼다. 어긋나면 LabelError."""
    out: dict[str, str] = {}
    for field in TEXT_FIELDS:
        value = changes.get(field, "")
        if not isinstance(value, str):
            raise LabelError(f"{field} 는 글자여야 합니다")
        out[field] = " ".join(value.split())  # 줄바꿈과 겹친 빈칸이 표에 들어가지 않게
    if out["fitness_factor"] and out["fitness_factor"] not in FACTORS:
        raise LabelError(f"체력 요인은 {', '.join(FACTORS)} 가운데 하나이거나 비워 둡니다")
    if out["phase"] not in PHASES:
        raise LabelError(f"단계는 {', '.join(PHASES)} 가운데 하나입니다")
    for field in FLAG_FIELDS:
        value = changes.get(field)
        if not isinstance(value, bool):
            raise LabelError(f"{field} 는 참 또는 거짓이어야 합니다")
        out[field] = str(value)
    return out


def save_label(path: Path, name: str, changes: Mapping[str, object]) -> dict[str, str]:
    """한 줄을 고쳐 표를 다시 쓴다. 고친 줄을 돌려준다.

    줄 순서와 다른 줄은 읽은 글자 그대로 둔다(video.labels.fill_factors 와 같은 방식).
    고친 것이 없어도 source 는 human 이 된다.
    """
    values = _checked(changes)
    fields, rows = read_labels(path)
    found = [row for row in rows if row["name_on_video"] == name]
    if not found:
        raise LabelError(f"표에 없는 이름입니다: {name}")
    row = found[0]
    after = {**row, **values, "source": "human", "score": HUMAN_SCORE}
    if after != row:
        row.update(after)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, lineterminator="\n", fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return dict(row)


def _seconds(value: object, what: str) -> int:
    # 참 · 거짓도 파이썬에서는 정수다. 0초 · 1초로 읽히지 않게 따로 막는다.
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelError(f"{what}은 초 단위 정수여야 합니다")
    if value < 0:
        raise LabelError(f"{what}은 0초보다 앞일 수 없습니다")
    return value


def save_clip_times(
    release_dir: Path, video_id: str, seq: int, start_sec: object, end_sec: object
) -> dict[str, Any]:
    """클립 하나의 시작 · 끝을 고친다. 고친 클립을 돌려준다.

    video_clips.csv 의 그 줄을 고치고(서비스가 읽는 표), clip_times.csv 에도 적는다
    (make clips 가 다시 덮어쓸 표). 다른 줄은 읽은 글자 그대로 둔다.
    """
    start = _seconds(start_sec, "시작")
    end = _seconds(end_sec, "끝")
    if end <= start:
        raise LabelError("끝은 시작보다 뒤여야 합니다")

    path = release_dir / "video_clips.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    found = [row for row in rows if row["video_id"] == video_id and row["seq"] == str(seq)]
    if not found:
        raise LabelError(f"표에 없는 클립입니다: {video_id} #{seq}")
    row = found[0]
    after = {
        **row,
        "start_sec": str(start),
        "end_sec": str(end),
        "duration_sec": str(end - start),
    }
    if after != row:
        row.update(after)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, lineterminator="\n", fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        kept = [
            edit
            for edit in clip_table.read_edits(release_dir)
            if (edit.video_id, edit.seq) != (video_id, seq)
        ]
        edit = clip_table.Edit(video_id, seq, row["name_on_video"], start, end)
        clip_table.write_edits(release_dir, [*kept, edit])
    return {"video_id": video_id, "seq": seq, "start_sec": start, "end_sec": end, "edited": True}


def _videos() -> dict[str, dict[str, str]]:
    """영상 id → 제목과 연령대. 코퍼스의 영상 청크에서 읽는다."""
    out: dict[str, dict[str, str]] = {}
    for chunk in corpus().chunks:
        if chunk.source != "video":
            continue
        _, _, title = chunk.citation_label.partition(" · ")
        out[chunk.chunk_id.split(":", 1)[1]] = {
            "title": video_title(title) or chunk.citation_label,
            "age_group": chunk.age_group,
        }
    return out


def state(release_dir: Path) -> dict[str, Any]:
    """화면이 그릴 것 전부. 라벨 줄마다 그 이름의 클립을 붙인다."""
    videos = _videos()
    edited = {(edit.video_id, edit.seq) for edit in clip_table.read_edits(release_dir)}
    clips: dict[str, list[dict[str, Any]]] = {}
    with (release_dir / "video_clips.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            video = videos.get(row["video_id"], {})
            clips.setdefault(row["name_on_video"], []).append(
                {
                    "video_id": row["video_id"],
                    "seq": int(row["seq"]),
                    "edited": (row["video_id"], int(row["seq"])) in edited,
                    "start_sec": int(row["start_sec"]),
                    "end_sec": int(row["end_sec"]),
                    "phase_on_video": row["phase_on_video"],
                    "title": video.get("title", ""),
                    "age_group": video.get("age_group", ""),
                }
            )
    _, rows = read_labels(release_dir / "clip_labels.csv")
    return {
        "factors": list(FACTORS),
        "phases": list(PHASES),
        "vocabulary": list(vocabulary.all_names()),
        "labels": [{**row, "clips": clips.get(row["name_on_video"], [])} for row in rows],
    }


class Handler(BaseHTTPRequestHandler):
    release_dir: Path

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 — http.server 가 정한 이름
        if self.path == "/":
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(HTTPStatus.OK, state(self.release_dir))
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "그런 경로가 없습니다"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/api/labels", "/api/clips"):
            self._json(HTTPStatus.NOT_FOUND, {"error": "그런 경로가 없습니다"})
            return
        # 표를 고치는 요청은 이 화면이 보낸 JSON 만 받는다. 다른 사이트가 브라우저를 시켜
        # 보내는 폼 요청은 이 머리말을 달 수 없다.
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON 으로 보내 주세요"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise LabelError("요청을 읽을 수 없습니다")
            saved = self._clip(body) if self.path == "/api/clips" else self._label(body)
        except (LabelError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._json(HTTPStatus.OK, saved)

    def _label(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body.get("name_on_video"), str):
            raise LabelError("name_on_video 가 없습니다")
        return save_label(self.release_dir / "clip_labels.csv", body["name_on_video"], body)

    def _clip(self, body: dict[str, Any]) -> dict[str, Any]:
        video_id, seq = body.get("video_id"), body.get("seq")
        if not isinstance(video_id, str) or isinstance(seq, bool) or not isinstance(seq, int):
            raise LabelError("video_id 와 seq 가 없습니다")
        return save_clip_times(
            self.release_dir, video_id, seq, body.get("start_sec"), body.get("end_sec")
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # 저장한 것만 한 줄씩 남긴다. 화면이 부르는 GET 까지 찍으면 터미널이 묻힌다.
        if self.command == "POST":
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="클립 라벨과 시작 · 끝 시각을 화면에서 보고 고친다"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="브라우저를 열지 않는다")
    args = parser.parse_args()

    Handler.release_dir = settings().release_dir
    # 밖에서 닿지 않게 이 PC 에만 연다. 인증이 없고 표를 고친다.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"클립 라벨 도구: {url}  (끝내려면 Ctrl+C)")
    print(
        f"고치는 표: {Handler.release_dir}/clip_labels.csv · video_clips.csv · {clip_table.EDITS}"
    )
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n끝냈습니다")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
