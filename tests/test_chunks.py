"""코퍼스 청크 (docs/dev/AI-8 §2 · docs/04 §1·§2).

원자료 없이 돈다 — CI 에는 원자료가 없다. 모양은 실제 산출물을 그대로 본뜬다.
"""

from __future__ import annotations

import pandas as pd

from family_fitness_ai.rag import chunks as K


def labels(**row: str) -> pd.DataFrame:
    blank = {
        "video_id": "v1",
        "title": "🦖유아기 운동체력 향상",
        "age_group": "유아기",
        "fitness_factors": "민첩성;협응력",
        "duration_sec": "1023",
        "playlist_titles": "🎯 연령별 맞춤 운동 : 영유아",
    }
    return pd.DataFrame([{**blank, **row}])


def exercises(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": "v1", "exercise_name": name, "start_sec": start, "common": common}
            for name, start, common in rows
        ],
        columns=["video_id", "exercise_name", "start_sec", "common"],
    )


def test_영상_한_편이_청크_하나다() -> None:
    got, skipped = K.video_chunks(
        labels(), exercises(("지그재그 달려요", "60", "False"), ("거북이 스트레칭", "152", "True"))
    )
    (chunk,) = got
    assert skipped == 0
    assert chunk["chunk_id"] == "video:v1"
    assert chunk["citation_label"] == "국민체력100 운동영상 · 🦖유아기 운동체력 향상"
    assert chunk["citation_url"] == "https://www.youtube.com/watch?v=v1"
    assert chunk["age_group"] == "유아기"
    assert "체력요인 민첩성·협응력" in chunk["text"]
    assert "나오는 운동: 지그재그 달려요(01:00)" in chunk["text"]
    # 공통 준비·마무리는 뒤에 따로 — 본운동과 섞이면 어느 영상이나 같은 문장이 된다
    assert "준비·마무리: 거북이 스트레칭(02:32)" in chunk["text"]
    assert "길이 17분" in chunk["text"]


def test_연령이_빈_영상은_청크를_만들지_않고_센다() -> None:
    """docs/04 §2.2 — 이 검사가 `age_group IS NULL` 을 연령 무관으로 쓰는 것을 안전하게 만든다."""
    got, skipped = K.video_chunks(labels(age_group=""), exercises())
    assert (got, skipped) == ([], 1)


def test_운동_이름이_없는_영상은_설명문_첫_줄을_붙인다() -> None:
    """50토큰 미만 청크는 검색 잡음이다 (docs/04 §2.1). 해시태그 줄은 상용구라 건너뛴다."""
    description = "#국민체력100 #유아운동\n성장기 아이의 코어를 잡아 주는 운동입니다."
    videos = pd.DataFrame([{"video_id": "v1", "description": description}])
    (chunk,), _ = K.video_chunks(labels(), exercises(), videos)
    assert "🎯 연령별 맞춤 운동 : 영유아" in chunk["text"]  # 부모 문맥
    assert chunk["text"].endswith("성장기 아이의 코어를 잡아 주는 운동입니다.")


def test_운동_이름이_있으면_설명문을_붙이지_않는다() -> None:
    videos = pd.DataFrame([{"video_id": "v1", "description": "설명문입니다"}])
    (chunk,), _ = K.video_chunks(labels(), exercises(("걷기", "30", "False")), videos)
    assert "설명문입니다" not in chunk["text"]
    assert "연령별 맞춤 운동" not in chunk["text"]  # 본운동이 있으면 부모 문맥이 필요 없다


def test_시각이_없는_운동은_이름만_적는다() -> None:
    (chunk,), _ = K.video_chunks(labels(), exercises(("걷기", "", "False")))
    assert "나오는 운동: 걷기 ·" in chunk["text"] or chunk["text"].endswith("나오는 운동: 걷기")


def thresholds() -> pd.DataFrame:
    rows = [
        ("유아기", "F", "48", "53", "개월", "012", "앉아윗몸앞으로굽히기", "cm", "1", "14.3"),
        ("유아기", "F", "48", "53", "개월", "012", "앉아윗몸앞으로굽히기", "cm", "2", "12.5"),
        ("유아기", "F", "54", "59", "개월", "012", "앉아윗몸앞으로굽히기", "cm", "1", "14.3"),
        ("성인", "M", "19", "24", "세", "028", "상대악력", "%", "1", "61.0"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "age_group",
            "sex",
            "age_lo",
            "age_hi",
            "age_unit",
            "item_code",
            "item_name",
            "unit",
            "grade",
            "threshold",
        ],
    )


def test_기준표는_항목과_연령구간과_성별로_묶는다() -> None:
    got = K.criteria_chunks(thresholds())
    assert [c["chunk_id"] for c in got] == ["criteria:012-유아기-F", "criteria:028-성인-M"]
    text = got[0]["text"]
    assert text.startswith("국민체력100 인증 기준 · 앉아윗몸앞으로굽히기(cm) · 유아기 여자: ")
    assert "48~53개월 1등급 14.3cm, 2등급 12.5cm" in text
    assert "54~59개월 1등급 14.3cm" in text


def test_기준표_청크는_연령_무관이고_요인을_싣는다() -> None:
    """청크의 연령 값은 누구에게 보여도 되는가이지 누구에 관한 자료인가가 아니다 (docs/04 §2.2)."""
    got = K.criteria_chunks(thresholds())
    assert [c["age_group"] for c in got] == ["", ""]
    assert [c["fitness_factors"] for c in got] == ["유연성", "근력"]


def test_처방_청크는_본문과_인용을_그대로_옮긴다() -> None:
    frame = pd.DataFrame(
        [
            {
                "chunk_id": "prescription:유소년-11-F-본운동",
                "text": "유소년 11세 여자 100명에게 처방된 본운동: 달리기(10%)",
                "citation_label": "국민체력100 운동처방 · 유소년 11세",
                "age_group": "유소년",
            },
            {"chunk_id": "x", "text": "t", "citation_label": "c", "age_group": ""},
        ]
    )
    got, skipped = K.prescription_chunks(frame)
    (chunk,) = got
    assert skipped == 1
    assert chunk["chunk_id"] == "prescription:유소년-11-F-본운동"
    assert chunk["source"] == "prescription"
    assert chunk["fitness_factors"] == ""  # 처방문에 요인이 없다 (AI-6 §4 ③)
