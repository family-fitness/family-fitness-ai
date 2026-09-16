"""공단 유튜브 채널의 영상 메타데이터와 자막을 모은다 (docs/02).

API 는 `channels.list` → `playlists.list` → `playlistItems.list` → `videos.list` 만
쓴다. `search.list` 는 일일 100회 제한이다 (docs/02 §3.2).

자막은 공식 API 로 받을 수 없다 — `captions.download` 는 영상 주인 권한이 필요하다.
`youtube-transcript-api` 로 받고, 몰아서 부르면 막히므로 한 편마다 쉰다.

응답과 자막은 JSONL 에 캐시한다. **재실행이 다시 부르지 않는다** (docs/02 §4).
새로 받으려면 캐시 파일을 지운다.

실행 (저장소 루트에서):
    python -m family_fitness_ai.labeling.collect --playlist <재생목록 ID>   # 시험 묶음
    python -m family_fitness_ai.labeling.collect                            # 채널 전체
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from youtube_transcript_api import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from ..common.settings import get_settings

CHANNEL_ID = "UCpjBiFyCh3f5bDU99Izt8Fw"  # 국민체력100 (@kspo100)
API = "https://www.googleapis.com/youtube/v3"
PAGE_SIZE = 50  # 한 번에 받을 수 있는 최대

CACHE_DIR = Path("data/raw/youtube")
API_CACHE = "api.jsonl"
CAPTION_CACHE = "captions.jsonl"
SCREEN_CACHE = "screen.jsonl"  # frames 가 쓰고 label 이 읽는다
VIDEOS_FILE = "videos.csv"

CAPTION_LANGUAGES = ("ko",)
CAPTION_DELAY_SEC = 1.0

VIDEO_COLUMNS = [
    "video_id",
    "title",
    "description",
    "tags",
    "playlist_ids",
    "playlist_titles",
    "duration_sec",
    "published_at",
    "caption",
]

Json = dict[str, Any]


class JsonlCache:
    """키 하나에 응답 하나. append-only 다 — 같은 키가 다시 쓰이면 뒤의 것이 이긴다."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, Json] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.rows[row["key"]] = row

    def get(self, key: str) -> Json | None:
        row = self.rows.get(key)
        return None if row is None else row["value"]

    def put(self, key: str, value: Json) -> None:
        # 받은 시각을 남긴다. 오래된 캐시를 가려 지울 수 있게.
        fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
        row = {"key": key, "fetched_at": fetched_at, "value": value}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows[key] = row

    def values(self) -> dict[str, Json]:
        return {key: row["value"] for key, row in self.rows.items()}


def cache_key(endpoint: str, params: dict[str, str]) -> str:
    """API 키는 들어가지 않는다 — 요청 직전에 붙인다. 키가 캐시 파일에 남지 않는다."""
    return f"{endpoint}?{urllib.parse.urlencode(sorted(params.items()))}"


def call(cache: JsonlCache, endpoint: str, params: dict[str, str], api_key: str) -> Json:
    key = cache_key(endpoint, params)
    hit = cache.get(key)
    if hit is not None:
        return hit
    url = f"{API}/{endpoint}?{urllib.parse.urlencode({**params, 'key': api_key})}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload: Json = json.load(resp)
    except urllib.error.HTTPError as e:
        # 원래 예외에는 URL 이 실리고 URL 에 키가 있다. 잇지 않는다 (docs/01 §5).
        raise RuntimeError(f"YouTube API {endpoint} 실패 · HTTP {e.code}") from None
    cache.put(key, payload)
    return payload


def pages(cache: JsonlCache, endpoint: str, params: dict[str, str], api_key: str) -> Iterator[Json]:
    token = ""
    while True:
        page = {**params, "maxResults": str(PAGE_SIZE)} | ({"pageToken": token} if token else {})
        payload = call(cache, endpoint, page, api_key)
        yield from payload.get("items", [])
        token = payload.get("nextPageToken", "")
        if not token:
            return


def collect_videos(
    cache: JsonlCache, api_key: str, playlist_ids: list[str] | None
) -> tuple[list[Json], dict[str, list[str]], dict[str, str]]:
    """영상 원문, 영상 → 재생목록, 재생목록 → 제목.

    재생목록을 지정하지 않으면 채널 전체다 — 업로드 목록까지 훑어 어느 재생목록에도
    없는 영상도 받는다. 업로드 목록은 소속으로 치지 않는다 (모든 영상이 속한다).
    """
    query = {"id": ",".join(playlist_ids)} if playlist_ids else {"channelId": CHANNEL_ID}
    titles = {
        p["id"]: p["snippet"]["title"]
        for p in pages(cache, "playlists", {"part": "snippet", **query}, api_key)
    }

    members: dict[str, list[str]] = defaultdict(list)
    for pid in titles:
        items = pages(
            cache, "playlistItems", {"part": "contentDetails", "playlistId": pid}, api_key
        )
        for item in items:
            members[item["contentDetails"]["videoId"]].append(pid)

    video_ids = set(members)
    if not playlist_ids:
        channel = call(cache, "channels", {"part": "contentDetails", "id": CHANNEL_ID}, api_key)
        uploads = channel["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        items = pages(
            cache, "playlistItems", {"part": "contentDetails", "playlistId": uploads}, api_key
        )
        video_ids.update(item["contentDetails"]["videoId"] for item in items)

    ordered = sorted(video_ids)  # 묶음이 같아야 캐시가 맞는다
    videos: list[Json] = []
    for i in range(0, len(ordered), PAGE_SIZE):
        batch = ",".join(ordered[i : i + PAGE_SIZE])
        params = {"part": "snippet,contentDetails", "id": batch}
        videos += call(cache, "videos", params, api_key)["items"]
    return videos, members, titles


def fetch_captions(
    cache: JsonlCache,
    video_ids: list[str],
    client: YouTubeTranscriptApi | None = None,
    delay_sec: float = CAPTION_DELAY_SEC,
) -> dict[str, Json]:
    """영상 → 자막. **자막이 없다는 것도 결과**라 캐시한다.

    YouTube 가 막으면 거기서 새로 묻기를 멈추고, 캐시에 있는 것까지만 돌려준다. 막힌
    영상은 캐시하지 않는다 — 나중에 다시 돌리면 끊긴 곳부터 받는다. 받지 못한 영상은
    결과에 없고, `videos.csv` 의 `caption` 이 빈칸으로 드러난다.
    """
    client = client or YouTubeTranscriptApi()
    out: dict[str, Json] = {}
    blocked = False
    for vid in video_ids:
        value = cache.get(vid)
        if value is None and not blocked:
            try:
                t = client.fetch(vid, languages=CAPTION_LANGUAGES)
            except RequestBlocked:
                blocked = True
                continue
            except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
                value = {"missing": type(e).__name__}
            else:
                value = {
                    "language": t.language_code,
                    "generated": t.is_generated,
                    "snippets": [{"start": s.start, "text": s.text} for s in t.snippets],
                }
            cache.put(vid, value)
            time.sleep(delay_sec)
        if value is not None:
            out[vid] = value
    return out


def caption_status(caption: Json | None) -> str:
    if caption is None:
        return ""
    if "missing" in caption:
        return str(caption["missing"])
    return "auto" if caption["generated"] else "manual"


_DURATION = re.compile(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?")


def parse_duration(iso: str) -> int | None:
    """`PT26M15S` → 1575. 모르는 모양이면 `None` — 0 으로 채우지 않는다."""
    m = _DURATION.fullmatch(iso)
    if m is None:
        return None
    days, hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def videos_frame(
    videos: list[Json],
    members: dict[str, list[str]],
    titles: dict[str, str],
    captions: dict[str, Json],
) -> pd.DataFrame:
    rows = []
    for v in videos:
        s = v["snippet"]
        playlist_ids = list(dict.fromkeys(members.get(v["id"], [])))
        rows.append(
            {
                "video_id": v["id"],
                "title": s["title"],
                "description": s.get("description", ""),
                "tags": ";".join(s.get("tags", [])),
                "playlist_ids": ";".join(playlist_ids),
                "playlist_titles": ";".join(titles[p] for p in playlist_ids),
                "duration_sec": parse_duration(v["contentDetails"]["duration"]),
                "published_at": s["publishedAt"],
                "caption": caption_status(captions.get(v["id"])),
            }
        )
    frame = pd.DataFrame(rows, columns=VIDEO_COLUMNS)
    frame["duration_sec"] = frame["duration_sec"].astype("Int64")  # 1575.0 으로 쓰이지 않게
    return frame.sort_values("video_id", ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="공단 유튜브 채널의 영상 메타데이터와 자막을 모은다")
    ap.add_argument(
        "--playlist", action="append", help="이 재생목록만 받는다 (여러 번). 없으면 채널 전체"
    )
    ap.add_argument("--cache", default=str(CACHE_DIR), help="API 응답·자막 캐시 (커밋하지 않는다)")
    ap.add_argument(
        "--interim", default="data/interim", help="videos.csv 를 낼 곳 (커밋하지 않는다)"
    )
    args = ap.parse_args(argv)

    api_key = get_settings().youtube_api_key
    if not api_key:
        print("[중단] YOUTUBE_API_KEY 가 비어 있다 (.env)")
        return 1

    cache_dir = Path(args.cache)
    videos, members, titles = collect_videos(
        JsonlCache(cache_dir / API_CACHE), api_key, args.playlist
    )
    captions = fetch_captions(
        JsonlCache(cache_dir / CAPTION_CACHE), sorted(v["id"] for v in videos)
    )
    frame = videos_frame(videos, members, titles, captions)

    interim = Path(args.interim)
    interim.mkdir(parents=True, exist_ok=True)
    # utf-8-sig — 검수하는 사람이 엑셀로 연다 (docs/02 §4)
    frame.to_csv(interim / VIDEOS_FILE, index=False, encoding="utf-8-sig")

    status = Counter(frame["caption"])
    print(f"재생목록 {len(titles)} · 영상 {len(frame):,} → {interim / VIDEOS_FILE}")
    print("자막 " + " · ".join(f"{k or '받지 않음'} {n}" for k, n in sorted(status.items())))
    if status[""]:
        print(f"[알림] 자막 {status['']}편을 받지 못했다 (YouTube 차단). 다시 돌리면 이어 받는다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
