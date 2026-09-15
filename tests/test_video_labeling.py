"""영상 수집과 라벨 규칙 (docs/dev/AI-7 §3).

네트워크 없이 돈다 — CI 에는 키가 없다. 제목·설명문은 공단 영상의 실제 모양을 본뜬다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from family_fitness_ai.labeling import collect as C
from family_fitness_ai.labeling import label as L

INFANTS = ("PLBdpvOnWjVZmeshOCp06HAov4ieoNHoen", "🎯 연령별 맞춤 운동 : 영유아")
MIDDLE_AGED = ("PLBdpvOnWjVZlsOa_LGjvlta7sg3NBFbqs", "🎯 연령별 맞춤 운동 : 중장년")

VOCAB = L.Vocabulary(
    ["거북이 스트레칭", "흔들어 체조", "공 받고 던져요", "줄넘기", "줄넘기 운동", "동물처럼 걸어요"]
)


def videos(*rows: dict[str, str]) -> pd.DataFrame:
    blank = dict.fromkeys(C.VIDEO_COLUMNS, "")
    return pd.DataFrame([{**blank, **r} for r in rows], columns=C.VIDEO_COLUMNS)


def in_playlists(*playlists: tuple[str, str]) -> dict[str, str]:
    return {
        "playlist_ids": ";".join(p[0] for p in playlists),
        "playlist_titles": ";".join(p[1] for p in playlists),
    }


def label_one(caption: dict[str, Any] | None = None, **row: str) -> L.VideoLabel:
    captions = {"v1": caption} if caption else {}
    (label,), _ = L.label_videos(videos({"video_id": "v1", **row}), captions, VOCAB)
    return label


def snippets(*texts: str) -> dict[str, Any]:
    return {
        "generated": True,
        "snippets": [{"start": float(i), "text": t} for i, t in enumerate(texts)],
    }


# ── 수집 ────────────────────────────────────────────────────────────


def test_길이는_초로_바꾸고_모르는_모양은_비운다() -> None:
    assert C.parse_duration("PT26M15S") == 1575
    assert C.parse_duration("PT1H2M3S") == 3723
    assert C.parse_duration("P0D") == 0
    assert C.parse_duration("모름") is None


def test_캐시에_있으면_API_를_부르지_않는다(tmp_path: Path) -> None:
    """재실행이 API 를 다시 때리지 않는다 (docs/02 §4). 키가 가짜여도 캐시에서 나온다."""
    path = tmp_path / "api.jsonl"
    payload = {"items": [{"id": "v1"}]}
    C.JsonlCache(path).put(C.cache_key("videos", {"part": "snippet", "id": "v1"}), payload)

    reopened = C.JsonlCache(path)  # 파일에서 다시 읽는다
    assert C.call(reopened, "videos", {"id": "v1", "part": "snippet"}, api_key="가짜") == payload


def test_수집_행은_재생목록_제목과_자막_상태를_싣는다() -> None:
    video = {
        "id": "v1",
        "snippet": {
            "title": "기초체력 향상을 위해",
            "description": "",
            "tags": ["국민체력100", "초등학생운동"],
            "publishedAt": "2024-11-18T00:00:00Z",
        },
        "contentDetails": {"duration": "PT11M52S"},
    }
    frame = C.videos_frame(
        [video],
        {"v1": [INFANTS[0], INFANTS[0]]},  # 같은 재생목록에 두 번 들어 있어도
        {INFANTS[0]: INFANTS[1]},
        {"v1": snippets("안녕하세요")},
    )
    row = frame.iloc[0]
    assert row["tags"] == "국민체력100;초등학생운동"
    assert row["playlist_titles"] == INFANTS[1]
    assert row["duration_sec"] == 712
    assert row["caption"] == "auto"


def test_자막이_없는_사유를_남긴다() -> None:
    assert C.caption_status({"missing": "TranscriptsDisabled"}) == "TranscriptsDisabled"
    assert C.caption_status(None) == ""


def test_막히면_새로_묻기를_멈추고_막힌_영상은_캐시하지_않는다(tmp_path: Path) -> None:
    """다시 돌리면 끊긴 곳부터 받는다 (docs/dev/AI-7 §2). 캐시에 있던 것은 그대로 싣는다."""

    class Snippet:
        start, text = 0.0, "거북이 스트레칭"

    class Transcript:
        language_code, is_generated, snippets = "ko", True, [Snippet()]

    class Client:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def fetch(self, video_id: str, languages: tuple[str, ...]) -> Transcript:
            self.asked.append(video_id)
            if video_id == "b":
                raise C.RequestBlocked(video_id)
            return Transcript()

    cache = C.JsonlCache(tmp_path / "captions.jsonl")
    cache.put("d", {"missing": "TranscriptsDisabled"})
    client = Client()

    got = C.fetch_captions(cache, ["a", "b", "c", "d"], client=client, delay_sec=0)  # type: ignore[arg-type]

    assert client.asked == ["a", "b"]  # 막힌 뒤로는 묻지 않는다
    assert set(got) == {"a", "d"}
    assert set(C.JsonlCache(cache.path).rows) == {"a", "d"}  # b 는 캐시되지 않았다


# ── 연령 ────────────────────────────────────────────────────────────


def test_재생목록에_적힌_연령을_쓴다() -> None:
    """`QU-eRtSLsm0` — 제목은 「유치원생」이고 재생목록이 「영유아」다."""
    label = label_one(title="🐣 유치원생도 쉽게 따라할 수 있는 운동", **in_playlists(INFANTS))
    assert label.age_group == "유아기"
    assert label.age_evidence == f"재생목록: {INFANTS[1]}"


def test_태그와_설명문의_연령은_옮기지_않는다() -> None:
    """태그는 검색 노출용이다 — 유치원생 영상에 `초등학생운동` 이 달려 있다."""
    label = label_one(
        title="유치원생도 따라할 수 있는 운동",
        tags="초등학생운동;영유아운동",
        description="청소년에게도 좋아요",
    )
    assert label.age_group is None
    assert label.age_evidence == ""


def test_제목에_적힌_연령이_재생목록보다_앞선다() -> None:
    """「청소년」 재생목록에 `[유소년]` 영상이 섞여 있다 (docs/dev/AI-7 §3)."""
    youth = ("PLBdpvOnWjVZl_rR4d70BQB8nens2vpbhk", "🎯 연령별 맞춤 운동 : 청소년")
    title = "[👦🏻유소년] 성장기 학생들을 위한 근력 운동 프로그램 (30min)"
    label = label_one(title=title, **in_playlists(youth))
    assert (label.age_group, label.age_evidence) == ("유소년", f"제목: {title}")
    # `초등학생` 은 연령대 이름이 아니다 — 제목에 연령이 없으니 재생목록을 쓴다
    assert (
        label_one(title="[초등학생] 키 쑥쑥 스트레칭", **in_playlists(youth)).age_group == "청소년"
    )


def test_한_곳에_서로_다른_연령이_적히면_비우고_근거는_남긴다() -> None:
    label = label_one(title="유아기와 청소년이 함께하는 운동", **in_playlists(INFANTS))
    assert label.age_group is None
    assert label.age_evidence == "제목: 유아기와 청소년이 함께하는 운동"


def test_재생목록의_청년은_성인이다() -> None:
    assert L.age_of("🎯 연령별 맞춤 운동 : 청년", [])[0] == "성인"
    assert L.age_of("🎯 연령별 맞춤 운동 : 청소년", [])[0] == "청소년"


def test_중장년_재생목록은_라벨하지_않는다() -> None:
    """서비스 대상이 아니다 (docs/dev/AI-7 §3.5)."""
    labels, excluded = L.label_videos(
        videos(
            {"video_id": "a", **in_playlists(INFANTS)},
            {"video_id": "b", **in_playlists(MIDDLE_AGED)},
        ),
        {},
        VOCAB,
    )
    assert [v.video_id for v in labels] == ["a"]
    assert excluded == 1


# ── 요인 ────────────────────────────────────────────────────────────


def test_요인은_제목에서_표기_차이까지만_합친다() -> None:
    assert L.factors_of("기초체력 향상 체조 | 심폐체력 유연성 운동")[0] == ("심폐지구력", "유연성")
    assert L.factors_of("소근육 발달 협응성 향상 운동")[0] == ("협응력",)
    assert L.factors_of("대근육 발달 근력/근지구력 운동")[0] == ("근력", "근지구력")


def test_균형은_요인으로_읽지_않는다() -> None:
    assert L.factors_of("균형있는 근육 발달을 위한 이동성 운동") == ((), "")


def test_설명문의_요인_해시태그는_보지_않는다() -> None:
    label = label_one(
        title="꽃게처럼 걸어요", description="#근력 #근지구력 #민첩성 #순발력 #협응성"
    )
    assert label.fitness_factors == ()


# ── 운동명 ──────────────────────────────────────────────────────────


def test_설명문의_해시태그는_운동명으로_읽지_않는다() -> None:
    """`#흔들어체조` 가 영유아 재생목록 설명문 대부분에 있다 (docs/dev/AI-7 §3.2)."""
    label = label_one(
        title="꽃게처럼 걸어요", description="#국민체력100 #헤이지니#유아운동 #흔들어체조"
    )
    assert label.exercises == ()


def test_띄어쓰기만_다른_이름을_잡고_그_줄을_근거로_남긴다() -> None:
    (match,) = label_one(
        title="응용운동", description="10:12 조심조심 걸어요\n11:48 공받고 던져요"
    ).exercises
    assert match == L.Match("공 받고 던져요", "description", "11:48 공받고 던져요")


def test_긴_이름이_짧은_이름으로_한_번_더_잡히지_않는다() -> None:
    label = label_one(title="줄넘기 운동 해요")
    assert [m.name for m in label.exercises] == ["줄넘기 운동"]


def test_자막_조각_경계에_걸친_이름을_잡는다() -> None:
    caption = snippets("마무리 거북이", "스트레칭 시작", "[음악]")
    (match,) = label_one(caption, title="꽃게처럼 걸어요").exercises
    assert match == L.Match("거북이 스트레칭", "captions", "마무리 거북이 스트레칭 시작")


def test_같은_이름은_제목_설명문_자막_순으로_하나만_남긴다() -> None:
    (match,) = label_one(
        snippets("동물처럼 걸어요"),
        title="EP01.동물처럼 걸어요 (20min)",
        description="동물처럼 걸어요",
    ).exercises
    assert match.source == "title"


def test_어휘_밖_이름은_만들지_않는다() -> None:
    assert label_one(title="수박나라로 떠나요!", description="00:16 로켓 발사해요").exercises == ()


# ── 산출 ────────────────────────────────────────────────────────────


def test_운동_이름_표는_믿을_만한_출처부터_적는다() -> None:
    labels, _ = L.label_videos(
        videos(
            {"video_id": "a", "title": "꽃게처럼 걸어요"},
            {"video_id": "b", "title": "EP03.거북이 스트레칭"},
        ),
        {"a": snippets("거북이 스트레칭")},
        VOCAB,
    )
    frame = L.exercises_frame(labels)
    assert list(frame["video_id"]) == ["b", "a"]
    assert list(frame["source"]) == ["title", "captions"]


def test_재지_않는_칸은_비워서_낸다() -> None:
    """지어내지 않는다 (AGENTS.md §4). 영상 단위 `start_sec` 은 비우고 운동별 표에 싣는다."""
    frame = L.labels_frame([label_one(title="유아기 운동")])
    assert (frame[["start_sec", "space", "noise", "equipment", "intensity"]] == "").all().all()
    assert frame.loc[0, "labeler_version"] == "labeler:rules/v3"


# ── 화면 이름표 ──────────────────────────────────────────────────────


HEADER = "(응용편) 민첩성·순발력·협응성"


def frame(t: int, *texts: str) -> dict[str, Any]:
    return {"t": t, "texts": list(texts)}


def spans(segments: list[L.Segment]) -> list[tuple[str, int, int]]:
    return [(s.name, s.start_sec, s.end_sec) for s in segments]


def test_이름표가_이어지는_동안이_한_구간이다() -> None:
    """이름표가 잠깐 사라져도(4초) 사이에 다른 이름이 없으면 잇는다."""
    segments = L.screen_segments(
        [
            frame(0, "100", HEADER, "빠르게 맞춰요"),
            frame(2, "100", HEADER, "빠르게 맞춰요"),
            frame(4, "100"),
            frame(6, HEADER, "빠르게 맞춰요"),
            frame(8, "(응용편) 민첩성·근지구력", "지그재그 달려요"),
            frame(10, "(응용편) 민첩성·근지구력", "지그재그 달려요"),
        ]
    )
    assert spans(segments) == [("빠르게 맞춰요", 0, 6), ("지그재그 달려요", 8, 10)]


def test_한_프레임짜리_이름은_잡음으로_버리고_앞뒤를_잇는다() -> None:
    segments = L.screen_segments(
        [
            frame(0, "후다닥 붙여요"),
            frame(2, "후다닥 붙여요"),
            frame(4, "후다닥 붙어요"),  # 한 프레임만 잘못 읽었다
            frame(6, "후다닥 붙여요"),
            frame(8, "후다닥 붙여요"),
        ]
    )
    assert spans(segments) == [("후다닥 붙여요", 0, 8)]


def test_괄호와_샵은_이름표_비교에서_뗀다() -> None:
    segments = L.screen_segments(
        [
            frame(0, "우리 모두 다 같이"),
            frame(2, "우리 모두 다 같이)"),
            frame(4, "#우리 모두 다 같이"),
        ]
    )
    assert spans(segments) == [("우리 모두 다 같이", 0, 4)]


def test_숫자와_영문_로고는_이름이_아니다() -> None:
    assert L.screen_segments([frame(0, "100", "KSPO"), frame(2, "100", "Ts")]) == []


def test_윗줄_요인은_여러_프레임에서_읽힌_것만_붙인다() -> None:
    """글자 인식이 `협응성` 을 `협성`·`형성` 으로 읽는 프레임이 있다. 다수결로 붙인다."""
    headers = [HEADER, HEADER, "(응용편) 민첩성·순발력·협성", HEADER, "(응용편) 민첩성·근력·형성"]
    (segment,) = L.screen_segments(
        [frame(i * 2, h, "빠르게 맞춰요") for i, h in enumerate(headers)]
    )
    assert segment.factors == ("민첩성", "순발력", "협응력")  # 근력은 한 프레임뿐이라 빠진다


def test_기본운동기술은_요인이_아니다() -> None:
    header = "(복합편) 기본운동기술 안정성·이동성·조작성"
    (segment,) = L.screen_segments([frame(t, header, "꽃게처럼 걸어요") for t in (0, 2, 4)])
    assert segment.factors == ()


def test_화면_이름표는_시작_시각과_요인을_싣는다() -> None:
    screen = {"frames": [frame(t, HEADER, "공받고 던져요") for t in (700, 702, 704)]}
    labels, _ = L.label_videos(
        videos({"video_id": "v1", "title": "응용운동"}), {}, VOCAB, {"v1": screen}
    )
    (match,) = labels[0].exercises
    assert match == L.Match(
        "공 받고 던져요", "screen", "11:40 공받고 던져요", 700, ("민첩성", "순발력", "협응력")
    )
    row = L.exercises_frame(labels).iloc[0]
    assert (row["start_sec"], row["fitness_factors"]) == (700, "민첩성;순발력;협응력")
    assert row["url"] == "https://www.youtube.com/watch?v=v1&t=700s"  # 그 운동부터 재생된다
    assert labels[0].screen == "1"


def test_시각이_없으면_영상_처음부터_재생되는_주소다() -> None:
    labels, _ = L.label_videos(
        videos({"video_id": "v1", "title": "EP03.거북이 스트레칭"}), {}, VOCAB
    )
    row = L.exercises_frame(labels).iloc[0]
    assert (row["start_sec"], row["url"]) == ("", "https://www.youtube.com/watch?v=v1")


def test_제목에서_잡힌_이름도_시각은_화면에서_가져온다() -> None:
    """제목이 더 믿을 만해도 시각은 화면에만 있다 (docs/dev/AI-7 §3.4)."""
    screen = {"frames": [frame(t, "거북이 스트레칭") for t in (30, 32)]}
    labels, _ = L.label_videos(
        videos({"video_id": "v1", "title": "EP03.거북이 스트레칭"}), {}, VOCAB, {"v1": screen}
    )
    (match,) = labels[0].exercises
    assert (match.source, match.start_sec) == ("title", 30)


def test_영상_요인은_제목과_화면을_합친다() -> None:
    screen = {"frames": [frame(t, "(응용편) 평형성", "밸런스 볼에 서요") for t in (0, 2)]}
    (label,), _ = L.label_videos(
        videos({"video_id": "v1", "title": "대근육 발달 근력 운동"}), {}, VOCAB, {"v1": screen}
    )
    assert label.fitness_factors == ("근력", "평형성")
    assert (
        label.factor_evidence == "제목: 대근육 발달 근력 운동 | 화면 00:00 밸런스 볼에 서요: 평형성"
    )


def test_받을_수_없던_영상은_사유를_남긴다() -> None:
    screens = {"v1": {"missing": "ERROR: [youtube] v1: Private video"}}
    (label,), _ = L.label_videos(videos({"video_id": "v1", "title": "운동"}), {}, VOCAB, screens)
    assert label.screen == "ERROR: [youtube] v1: Private video"
    assert L.labels_frame([label_one(title="운동")]).loc[0, "screen"] == ""  # 안 읽은 영상


# ── 아래 이름 막대 ───────────────────────────────────────────────────


BAR_VOCAB = L.Vocabulary(
    [
        "다리뻗어 상체 숙이기",
        "나비자세",
        "힘차게 던져요",
        "팔굽혀펴기",
        "사이드 스텝",
        "공을 옮겨요(옆)",
    ]
)


def segment(name: str, start: int = 0) -> L.Segment:
    return L.Segment(name, start, start + 10, ())


def test_이름_막대는_어휘와_통째로_같을_때만_붙인다() -> None:
    """같은 자리에 설명 문장도 뜬다 — 부분 일치를 허용하면 문장 속 낱말이 이름이 된다 (§3.9)."""
    got = L.match_bar(
        BAR_VOCAB,
        [
            segment("다리 뻗어 상체 숙이기"),
            segment("책상에서 팔굽혀펴기"),
            segment("사이드 스텝으로 활동한다"),
        ],
    )
    assert [m.name for m in got] == ["다리뻗어 상체 숙이기"]


def test_이름_막대의_번호와_느낌표를_떼고_괄호_속_이름도_본다() -> None:
    got = L.match_bar(
        BAR_VOCAB,
        [
            segment("3. 힘차게 던져요!!"),
            segment("넙다리 안쪽 늘리기 (나비자세)"),
            segment("공을 옮겨요(옆)"),  # 괄호까지가 이름이면 통째로 붙고 괄호 속은 보지 않는다
        ],
    )
    assert [m.name for m in got] == ["힘차게 던져요", "나비자세", "공을 옮겨요(옆)"]


def test_이름_막대는_시각을_싣고_이름표보다_뒤에_믿는다() -> None:
    frames = [
        {"t": t, "texts": ["거북이 스트레칭"], "bar": ["설명 한 줄", "거북이 스트레칭"]}
        for t in (40, 42)
    ]
    frames += [{"t": t, "texts": [], "bar": ["공 받고 던져요"]} for t in (90, 92)]
    labels, _ = L.label_videos(
        videos({"video_id": "v1", "title": "운동"}), {}, VOCAB, {"v1": {"frames": frames}}
    )
    got = {m.name: (m.source, m.start_sec) for m in labels[0].exercises}
    assert got == {"거북이 스트레칭": ("screen", 40), "공 받고 던져요": ("screen_bar", 90)}
