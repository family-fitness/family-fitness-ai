"""클립 라벨을 화면에서 고치는 도구(src/script/label_clips.py). 표를 고쳐 쓰는 쪽만 본다."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest
from conftest import needs_index, needs_release

from family_fitness_ai.common.settings import settings

HEADER = (
    "name_on_video,exercise_name,fitness_factor,phase,is_exercise,home_ok,quiet,needs_props,"
    "source,score\n"
)
#: 실제 표에서 잘라 온 세 줄. 쉼표가 든 이름은 따옴표로 싸여 있다.
ROWS = (
    "가볍게 달리기,조깅,심폐지구력,준비운동,True,True,False,False,human,0.0\n"
    '"어깨, 등 스트레칭",,유연성,정리운동,True,True,True,False,llm,0.0\n'
    "캐치볼을 해요,,,본운동,True,True,True,True,llm,0.62\n"
)


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "src" / "script" / "label_clips.py"
    spec = importlib.util.spec_from_file_location("label_clips", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


@pytest.fixture
def table(tmp_path: Path) -> Path:
    path = tmp_path / "clip_labels.csv"
    path.write_text(HEADER + ROWS, encoding="utf-8")
    return path


def _changes(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "exercise_name": "",
        "fitness_factor": "협응력",
        "phase": "본운동",
        "is_exercise": True,
        "home_ok": True,
        "quiet": True,
        "needs_props": True,
    }
    return {**base, **over}


def test_a_saved_row_becomes_a_human_row(table: Path):
    row = tool.save_label(table, "캐치볼을 해요", _changes(exercise_name="공 양손으로 주고 받기"))
    assert row["exercise_name"] == "공 양손으로 주고 받기"
    assert row["fitness_factor"] == "협응력"
    assert (row["source"], row["score"]) == ("human", "0.0")


def test_the_other_rows_are_left_as_they_were_read(table: Path):
    tool.save_label(table, "캐치볼을 해요", _changes())
    lines = table.read_text(encoding="utf-8").splitlines(keepends=True)
    assert lines[:3] == (HEADER + ROWS).splitlines(keepends=True)[:3]
    assert lines[3] == "캐치볼을 해요,,협응력,본운동,True,True,True,True,human,0.0\n"


def test_saving_without_a_change_still_marks_the_row_as_checked(table: Path):
    same = _changes(fitness_factor="유연성", phase="정리운동", needs_props=False)
    row = tool.save_label(table, "어깨, 등 스트레칭", same)
    assert row["source"] == "human"
    assert '"어깨, 등 스트레칭",,유연성,정리운동,True,True,True,False,human,0.0\n' in (
        table.read_text(encoding="utf-8")
    )


def test_a_row_already_checked_and_unchanged_is_not_rewritten(table: Path):
    before = table.stat().st_mtime_ns
    same = _changes(
        exercise_name="조깅",
        fitness_factor="심폐지구력",
        phase="준비운동",
        quiet=False,
        needs_props=False,
    )
    tool.save_label(table, "가볍게 달리기", same)
    assert table.read_text(encoding="utf-8") == HEADER + ROWS
    assert table.stat().st_mtime_ns == before


def test_a_line_break_in_a_name_does_not_reach_the_table(table: Path):
    row = tool.save_label(table, "캐치볼을 해요", _changes(exercise_name=" 공  주고\n받기 "))
    assert row["exercise_name"] == "공 주고 받기"
    assert len(table.read_text(encoding="utf-8").splitlines()) == 4


@pytest.mark.parametrize(
    "bad",
    [
        {"fitness_factor": "체력"},  # 여덟 요인에 없는 말
        {"phase": "마무리운동"},  # 편성이 모르는 단계는 어디에도 안 뽑힌다
        {"phase": ""},
        {"quiet": "True"},  # 글자 「True」 는 참이 아니다
        {"exercise_name": None},
    ],
)
def test_a_value_the_table_cannot_hold_is_refused(table: Path, bad: dict[str, object]):
    with pytest.raises(tool.LabelError):
        tool.save_label(table, "캐치볼을 해요", _changes(**bad))
    assert table.read_text(encoding="utf-8") == HEADER + ROWS


def test_a_name_that_is_not_in_the_table_is_refused(table: Path):
    with pytest.raises(tool.LabelError):
        tool.save_label(table, "없는 이름", _changes())


@needs_release
def test_the_real_table_survives_a_read_and_a_write(tmp_path: Path):
    # 고친 줄만 달라져야 diff 가 읽힌다. 실제 표를 읽고 그대로 쓰면 한 글자도 달라지지 않는다.
    real = settings().release_dir / "clip_labels.csv"
    copy = tmp_path / "clip_labels.csv"
    copy.write_bytes(real.read_bytes())
    fields, rows = tool.read_labels(copy)
    human = next(row for row in rows if row["source"] == "human")
    unchanged = {
        **{field: human[field] for field in tool.TEXT_FIELDS},
        **{field: human[field] == "True" for field in tool.FLAG_FIELDS},
    }
    tool.save_label(copy, human["name_on_video"], unchanged)
    assert copy.read_bytes() == real.read_bytes()

    other = next(row for row in rows if row["source"] != "human")
    tool.save_label(copy, other["name_on_video"], {**unchanged, "phase": other["phase"]})
    before = real.read_text(encoding="utf-8").splitlines()
    after = copy.read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    assert sum(a != b for a, b in zip(before, after, strict=True)) == 1


@needs_release
@needs_index
def test_every_label_comes_with_its_clips():
    state = tool.state(settings().release_dir)
    assert state["phases"] == ["준비운동", "본운동", "정리운동"]
    assert "유연성" in state["factors"]
    names = [row["name_on_video"] for row in state["labels"]]
    assert len(names) == len(set(names))  # 이름이 겹치면 어느 줄을 고칠지 알 수 없다
    clips = [clip for row in state["labels"] for clip in row["clips"]]
    assert clips and all(clip["end_sec"] > clip["start_sec"] for clip in clips)
    assert all(clip["title"] and clip["age_group"] for clip in clips)


CLIPS = (
    "video_id,seq,name_on_video,phase_on_video,start_sec,end_sec,duration_sec\n"
    "-EATykJOvBQ,1,거북이 스트레칭,,6,78,72\n"
    "-EATykJOvBQ,2,캐치볼을 해요,본운동,78,140,62\n"
    'Eg3GpTv7z8s,1,"어깨, 등 스트레칭",준비운동,30,96,66\n'
)


@pytest.fixture
def release(table: Path) -> Path:
    """라벨 표 옆에 클립 표를 둔 release 폴더."""
    (table.parent / "video_clips.csv").write_text(CLIPS, encoding="utf-8")
    return table.parent


def test_a_saved_time_reaches_the_clip_table(release: Path):
    saved = tool.save_clip_times(release, "-EATykJOvBQ", 2, 80, 132)
    assert (saved["start_sec"], saved["end_sec"], saved["edited"]) == (80, 132, True)
    lines = (release / "video_clips.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    before = CLIPS.splitlines(keepends=True)
    # 길이도 같이 고친다. 다른 줄은 읽은 글자 그대로다.
    assert lines[2] == "-EATykJOvBQ,2,캐치볼을 해요,본운동,80,132,52\n"
    assert [lines[0], lines[1], lines[3]] == [before[0], before[1], before[3]]


def test_a_saved_time_is_also_kept_where_make_clips_will_find_it(release: Path):
    # video_clips.csv 는 make clips 가 새로 쓴다. 거기에만 적으면 사람이 고친 값이 사라진다.
    tool.save_clip_times(release, "Eg3GpTv7z8s", 1, 34, 90)
    tool.save_clip_times(release, "-EATykJOvBQ", 2, 80, 132)
    tool.save_clip_times(release, "Eg3GpTv7z8s", 1, 36, 92)  # 다시 고치면 앞의 것을 갈아 끼운다
    assert (release / "clip_times.csv").read_text(encoding="utf-8") == (
        "video_id,seq,name_on_video,start_sec,end_sec\n"
        "-EATykJOvBQ,2,캐치볼을 해요,80,132\n"
        'Eg3GpTv7z8s,1,"어깨, 등 스트레칭",36,92\n'
    )


def test_saving_the_time_it_already_has_writes_nothing(release: Path):
    tool.save_clip_times(release, "-EATykJOvBQ", 1, 6, 78)
    assert (release / "video_clips.csv").read_text(encoding="utf-8") == CLIPS
    assert not (release / "clip_times.csv").exists()


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (90, 90),  # 길이가 없다
        (100, 90),  # 끝이 시작보다 앞이다
        (-1, 30),
        (10.5, 30),  # 초 단위 정수만
        ("10", 30),
        (True, 30),  # 참은 1초가 아니다
        (10, None),
    ],
)
def test_a_time_that_makes_no_clip_is_refused(release: Path, start: object, end: object):
    with pytest.raises(tool.LabelError):
        tool.save_clip_times(release, "-EATykJOvBQ", 2, start, end)
    assert (release / "video_clips.csv").read_text(encoding="utf-8") == CLIPS
    assert not (release / "clip_times.csv").exists()


def test_a_clip_that_is_not_in_the_table_is_refused(release: Path):
    with pytest.raises(tool.LabelError):
        tool.save_clip_times(release, "-EATykJOvBQ", 9, 10, 30)


@pytest.fixture
def served(table: Path, monkeypatch: pytest.MonkeyPatch):
    """도구를 빈 포트에 띄운다. 화면이 부르는 대로 불러 본다."""
    monkeypatch.setattr(tool.Handler, "release_dir", table.parent, raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), tool.Handler)
    # 끝낼 때 serve_forever 가 다음 확인까지 기다린다. 기본 0.5초면 시험마다 그만큼 늘어진다.
    thread = threading.Thread(target=server.serve_forever, args=(0.02,), daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url: str, body: dict[str, object], content_type: str) -> tuple[int, dict[str, str]]:
    request = urllib.request.Request(
        url + "/api/labels",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_saves_through_the_server(served: str, table: Path):
    status, row = _post(
        served, {"name_on_video": "캐치볼을 해요", **_changes()}, "application/json"
    )
    assert status == 200
    assert row["source"] == "human"
    assert "캐치볼을 해요,,협응력,본운동,True,True,True,True,human,0.0\n" in table.read_text(
        encoding="utf-8"
    )


def test_a_bad_label_comes_back_as_a_message_and_changes_nothing(served: str, table: Path):
    body = {"name_on_video": "캐치볼을 해요", **_changes(phase="마무리운동")}
    status, answer = _post(served, body, "application/json")
    assert status == 400
    assert "단계" in answer["error"]
    assert table.read_text(encoding="utf-8") == HEADER + ROWS


def test_only_json_from_the_page_may_change_the_table(served: str, table: Path):
    # 다른 사이트가 브라우저를 시켜 보낼 수 있는 것은 폼이나 text/plain 뿐이다.
    status, _ = _post(served, {"name_on_video": "캐치볼을 해요", **_changes()}, "text/plain")
    assert status == 415
    assert table.read_text(encoding="utf-8") == HEADER + ROWS


def _post_to(url: str, path: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_saves_a_time_through_the_server(served: str, release: Path):
    body: dict[str, object] = {"video_id": "-EATykJOvBQ", "seq": 2, "start_sec": 80, "end_sec": 132}
    status, saved = _post_to(served, "/api/clips", body)
    assert status == 200
    assert saved == {**body, "edited": True}
    assert "-EATykJOvBQ,2,캐치볼을 해요,본운동,80,132,52\n" in (
        release / "video_clips.csv"
    ).read_text(encoding="utf-8")


def test_a_bad_time_comes_back_as_a_message(served: str, release: Path):
    body: dict[str, object] = {"video_id": "-EATykJOvBQ", "seq": 2, "start_sec": 140, "end_sec": 78}
    status, answer = _post_to(served, "/api/clips", body)
    assert status == 400
    assert "끝" in str(answer["error"])
    assert (release / "video_clips.csv").read_text(encoding="utf-8") == CLIPS


def test_the_page_itself_is_served(served: str):
    with urllib.request.urlopen(served + "/", timeout=5) as response:
        page = response.read().decode("utf-8")
    assert "클립 라벨 고치기" in page
