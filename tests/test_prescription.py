"""처방 어휘와 처방 청크 (docs/dev/AI-6 §7).

원자료 없이 돈다 — CI 에는 원자료가 없다. 모양은 실제 처방문을 그대로 본뜬다.
"""

from __future__ import annotations

import pandas as pd

from family_fitness_ai.ingest import measurements as M
from family_fitness_ai.rag import prescription as P


def frame(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[M.AGE_GROUP_COL, M.AGE_COL, M.SEX_COL, M.PRESCRIPTION_COL])


def many(age_group: str, age: int, sex: str, text: str, n: int = P.MIN_ROWS) -> list:
    return [(age_group, age, sex, text)] * n


TEXT = (
    "준비운동:팔굽혀펴기,종아리 스트레칭"
    " / 본운동:왕복달리기,가슴/어깨 스트레칭"
    " / 정리운동:목 굽힘/ 폄 I"
)


# ── 파싱 ────────────────────────────────────────────────────────────


def test_단계는_세_가지로_갈린다() -> None:
    parsed = P.parse(TEXT)
    assert list(parsed) == ["준비운동", "본운동", "정리운동"]


def test_이름_안의_슬래시가_단계_구분자로_쪼개지지_않는다() -> None:
    """`가슴/어깨 스트레칭`·`목 굽힘/ 폄 I` — 맨 `/` 로 가르면 이름이 쪼개진다."""
    parsed = P.parse(TEXT)
    assert parsed["본운동"] == ["왕복달리기", "가슴/어깨 스트레칭"]
    assert parsed["정리운동"] == ["목 굽힘/ 폄 I"]


def test_모르는_단계_라벨은_버린다() -> None:
    assert P.parse("준비운동:팔굽혀펴기 / 이상한단계:뭔가") == {"준비운동": ["팔굽혀펴기"]}


# ── 정규화 ──────────────────────────────────────────────────────────


def test_표기_차이는_합친다() -> None:
    """실측에서 합쳐진 넷은 전부 띄어쓰기다 (docs/dev/AI-6 §4 ②)."""
    assert P.identity("팔굽혀펴기") == P.identity("팔굽혀 펴기")
    assert P.identity("윗몸 말아 올리기") == P.identity("윗몸말아올리기")


def test_번호_단계_방향이_다르면_합치지_않는다() -> None:
    """비슷해 보이는 이름은 대부분 다른 운동이다. 유사도로 합치면 틀린다."""
    assert P.identity("대퇴이두근 스트레칭") != P.identity("대퇴이두근 스트레칭2")
    assert P.identity("누워서 배가로근 수축 I") != P.identity("누워서 배가로근 수축 II")
    assert P.identity("의자 잡고 전방으로 무릎 굽혀 들기") != P.identity(
        "의자 잡고 후방으로 무릎 굽혀 들기"
    )


def test_대표_표기는_가장_많이_쓰인_원문이다() -> None:
    df = frame(
        many("유소년", 11, "F", "준비운동:팔굽혀펴기", 5)
        + many("유소년", 11, "F", "준비운동:팔굽혀 펴기", 2)
    )
    (term,) = P.build_vocabulary(df)
    assert term.name == "팔굽혀펴기"
    assert term.raw_forms == ("팔굽혀 펴기", "팔굽혀펴기")  # 원본 문자열을 버리지 않는다
    assert term.count == 7


# ── 청크 ────────────────────────────────────────────────────────────


def build(rows: list) -> tuple[list[P.Chunk], list]:
    df = frame(rows)
    return P.build_chunks(df, P.build_vocabulary(df))


def test_청크는_칸과_단계_하나다() -> None:
    chunks, _ = build(many("유소년", 11, "F", TEXT))
    assert [c.chunk_id for c in chunks] == [
        "prescription:유소년-11-F-준비운동",
        "prescription:유소년-11-F-본운동",
        "prescription:유소년-11-F-정리운동",
    ]


def test_같은_원자료면_같은_chunk_id_와_같은_본문이_나온다() -> None:
    """저장소 제약이 없어 이 검사가 결정성의 유일한 방어다 (docs/dev/AI-8 §1.1).

    행 순서가 바뀌어도 같아야 한다 — 파일을 읽는 순서는 보장되지 않는다.
    """
    rows = many("유소년", 11, "F", TEXT) + many("성인", 41, "M", "본운동:달리기,팔굽혀펴기")
    first, _ = build(rows)
    second, _ = build(list(reversed(rows)))
    assert [(c.chunk_id, c.text) for c in first] == [(c.chunk_id, c.text) for c in second]


def test_빈도가_바뀌어도_chunk_id_는_그대로다() -> None:
    """원자료가 쌓이면 빈도가 바뀐다. 해시였다면 id 가 바뀌어 인용이 끊겼다."""
    before, _ = build(many("유소년", 11, "F", "본운동:왕복달리기"))
    after, _ = build(
        many("유소년", 11, "F", "본운동:왕복달리기", 40)
        + many("유소년", 11, "F", "본운동:팔굽혀펴기", 40)
    )
    assert before[0].chunk_id == after[0].chunk_id
    assert before[0].text != after[0].text  # 본문은 새 빈도를 따른다 (upsert)


def test_연령대가_빈_청크가_없다() -> None:
    """이 검사가 `age_group = ? OR age_group IS NULL` 을 안전하게 만든다 (docs/04 §2.2)."""
    chunks, _ = build(many("유소년", 11, "F", TEXT) + many("어르신", 70, "M", TEXT))
    assert all(c.age_group for c in chunks)


def test_유아기_인용은_개월이다() -> None:
    chunks, _ = build(many("유아기", 60, "M", "준비운동:거북이 스트레칭"))
    assert chunks[0].citation_label == "국민체력100 운동처방 · 유아기 60개월"
    assert chunks[0].age_unit == "개월"


def test_표본_30_미만인_칸은_청크를_만들지_않는다() -> None:
    """30명도 안 되는 칸의 "많이 처방된 운동" 은 우연과 구분되지 않는다."""
    chunks, skipped = build(
        many("어르신", 70, "F", TEXT) + many("어르신", 95, "M", TEXT, P.MIN_ROWS - 1)
    )
    assert {c.age for c in chunks} == {70}
    assert skipped == [("어르신", 95, "M", P.MIN_ROWS - 1)]


def test_본문은_누적_80퍼센트를_덮을_때까지_싣는다() -> None:
    rows = (
        many("성인", 41, "F", "본운동:A", 60)
        + many("성인", 41, "F", "본운동:B", 25)
        + many("성인", 41, "F", "본운동:C", 10)
        + many("성인", 41, "F", "본운동:D", 5)
    )
    (chunk,) = build(rows)[0]
    assert chunk.exercise_names == ("A", "B")  # 60% → 85% 에서 멈춘다
    assert "A(60%)" in chunk.text


def test_본문에_최대_40개까지만_싣는다() -> None:
    names = ",".join(f"운동{i:02d}" for i in range(60))  # 고르게 흩어져 80% 에 늦게 닿는다
    (chunk,) = build(many("성인", 41, "F", f"본운동:{names}"))[0]
    assert len(chunk.exercise_names) == P.MAX_EXERCISES


# ── 적재 ────────────────────────────────────────────────────────────


def test_적재는_어르신을_남기고_오류_나이를_거른다(tmp_path) -> None:  # noqa: ANN001
    """어르신은 점수를 내지 않을 뿐 질의응답은 정상 동작한다 (docs/02 §6 ⑦).

    `load_dir` 는 어르신을 빼지만 처방 코퍼스는 어르신도 쓴다.
    """
    pd.DataFrame(
        {
            M.AGE_GROUP_COL: ["어르신", "어르신", "유소년", "성인"],
            M.AGE_COL: [70, 984, 11, 41],  # 984 는 측정 오류 (docs/dev/AI-6 §6.2)
            M.SEX_COL: ["F", "M", "F", "M"],
            M.PRESCRIPTION_COL: ["본운동:걷기", "본운동:걷기", "본운동:달리기", None],
        }
    ).to_csv(tmp_path / "202607.csv", index=False, encoding="utf-8-sig")

    loaded = M.load_prescriptions(tmp_path)
    assert list(loaded[M.AGE_GROUP_COL]) == ["어르신", "유소년"]  # 오류 나이·빈 처방 제외
    assert list(loaded[M.AGE_COL]) == [70, 11]
