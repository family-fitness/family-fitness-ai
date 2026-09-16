"""화면 이름표 읽기 (docs/02).

YouTube 에 닿지 않고 돈다 — 파이프를 흉내 낸다. 글자 인식(맥 Vision)은 CI 에 없어 여기서
재지 않는다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from family_fitness_ai.labeling import frames as F

STREAM = "https://rr1---sn.googlevideo.com/videoplayback?sig=SECRET"


def piping(
    source_code: int = 0, source_err: str = "", sink_code: int = 0, sink_err: str = ""
) -> tuple[F.Pipe, list[tuple[list[str], list[str]]]]:
    """ffmpeg 는 실패해도 쓴 만큼은 프레임을 남긴다 — 그래서 늘 세 장을 쓴다."""
    calls: list[tuple[list[str], list[str]]] = []

    def pipe(source: list[str], sink: list[str]) -> F.PipeResult:
        calls.append((source, sink))
        for i in (1, 2, 3):
            Path(sink[-1].replace("%05d", f"{i:05d}")).write_bytes(b"jpg")
        return source_code, source_err, sink_code, sink_err

    return pipe, calls


def test_영상_파일을_쓰지_않고_파이프로_프레임만_뽑는다(tmp_path: Path) -> None:
    """yt-dlp 는 표준출력으로 보내고(`-o -`), ffmpeg 는 파이프를 읽는다.

    프레임 번호에서 시각을 낸다.
    """
    pipe, calls = piping()

    got = F.stream_frames("x", tmp_path / "frames" / "x", pipe=pipe)

    assert [t for t, _ in got] == [0, 2, 4]
    ((source, sink),) = calls
    assert source[source.index("-o") + 1] == "-"
    assert sink[sink.index("-i") + 1] == "pipe:0"
    assert [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix != ".jpg"] == []


def test_막히면_Blocked_를_올리고_프레임을_남기지_않는다(tmp_path: Path) -> None:
    """캐시하지 않고 멈춘다 — 다시 돌리면 이어서 읽는다."""
    pipe, _ = piping(
        source_code=1, source_err="ERROR: [youtube] x: Sign in to confirm you’re not a bot"
    )
    with pytest.raises(F.Blocked):
        F.stream_frames("x", tmp_path / "x", pipe=pipe)
    assert not (tmp_path / "x").exists()


def test_네트워크가_끊겨도_캐시하지_않고_멈춘다(tmp_path: Path) -> None:
    """영상이 없는 것과 닿지 못한 것은 다르다. 뒤의 것을 캐시하면 다시 돌려도 건너뛴다."""
    for stderr in (
        "ERROR: [youtube] x: Unable to download API page: Failed to resolve 'www.youtube.com'",
        "ERROR:",  # 사유를 알 수 없을 때도
    ):
        pipe, _ = piping(source_code=1, source_err=stderr)
        with pytest.raises(F.Blocked):
            F.stream_frames("x", tmp_path / "x", pipe=pipe)


def test_없는_영상은_사유를_올린다(tmp_path: Path) -> None:
    pipe, _ = piping(source_code=1, source_err="WARNING: 느리다\nERROR: [youtube] x: Private video")
    with pytest.raises(F.DownloadFailed, match="Private video"):
        F.stream_frames("x", tmp_path / "x", pipe=pipe)


def test_ffmpeg_가_실패하면_반쪽_프레임을_남기지_않고_멈춘다(tmp_path: Path) -> None:
    pipe, _ = piping(sink_code=1, sink_err=f"{STREAM}: Invalid data found when processing input")

    with pytest.raises(F.Blocked) as caught:
        F.stream_frames("x", tmp_path / "x", pipe=pipe)

    assert not (tmp_path / "x").exists()
    assert "SECRET" not in str(caught.value)  # 서명이 든 주소를 남기지 않는다


def test_파이프는_보내는_쪽_오류와_종료_코드를_따로_돌려준다() -> None:
    """네트워크 없이 실제 파이프를 돌린다. 받는 쪽이 끝까지 읽어야 보내는 쪽이 멈추지 않는다."""
    reader = [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"]
    loud = [sys.executable, "-c", "import sys; sys.stdout.write('x' * 300000)"]
    assert F._pipe(loud, reader) == (0, "", 0, "")

    broken = [sys.executable, "-c", "import sys; sys.stderr.write('ERROR: nope'); sys.exit(1)"]
    source_code, source_err, sink_code, _ = F._pipe(broken, reader)
    assert (source_code, source_err, sink_code) == (1, "ERROR: nope", 0)


def test_30일_지난_프레임만_지운다(tmp_path: Path) -> None:
    """docs/01 §4.3 — 재배포 금지를 운영으로 옮긴 수명이다."""
    day = 86400
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    os.utime(old, (0, 0))
    os.utime(new, (35 * day, 35 * day))

    assert F.purge_old_frames(tmp_path, now=40 * day) == 1
    assert not old.exists()
    assert new.exists()


def test_읽는_자리가_늘면_다시_읽고_없는_영상은_다시_보지_않는다() -> None:
    """이름 막대(`bar`)를 읽기 전에 캐시된 영상은 다시 읽는다 — 디스크의 프레임에서."""
    assert F.needs_reading(None)
    assert F.needs_reading({"frames": [{"t": 0, "texts": ["거북이 스트레칭"]}]})
    assert not F.needs_reading({"frames": [{"t": 0, "texts": [], "bar": []}]})
    assert not F.needs_reading({"missing": "ERROR: [youtube] x: Private video"})
