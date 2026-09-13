"""처방 어휘와 처방 청크 (docs/dev/AI-6).

원문은 측정 데이터의 처방 컬럼이다. 한 행이 이렇게 생겼다:

    준비운동:A,B,C / 본운동:D,E / 정리운동:F,G

**처방문 1건을 청크 1개로 두지 않는다.** 고유 처방문이 30만 개이고 81%가 한 번만
나온다 — 641개 어휘에서 뽑은 조합이라 서로 조금씩만 다른 사본이다. 청크는
`(연령구간, 나이, 성별, 단계)` 한 칸이고, 그 칸에서 처방된 운동을 빈도순으로 센 것이
본문이다 (docs/04 §1.1).

등급으로 한 번 더 나눈 층도 함께 낸다. 등급을 모르거나 그 등급 칸이 비면 등급 없는
층으로 떨어진다.

실행 (저장소 루트에서):
    python -m family_fitness_ai.rag.prescription --data-dir <원자료 디렉터리>
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..common.types import age_unit_of
from ..ingest import measurements as M
from ..stats.distribution import fold_grades
from ..stats.grade import GRADE_NAMES, PARTICIPATED

SOURCE = "prescription"
PHASES = ("준비운동", "본운동", "정리운동")

# 등급 층의 등급. 4·5·6등급은 참가로 접는다 — 계약의 등급은 넷이다 (docs/03 §3.4).
GRADES = (*GRADE_NAMES.values(), PARTICIPATED)

# 단계 구분자는 앞뒤 공백이 있는 ` / ` 다. 운동명 안에 `/` 가 들어 있어
# (`가슴/어깨 스트레칭`, `목 굽힘/ 폄 I`) 맨 `/` 로 가르면 이름이 쪼개진다.
PHASE_SEPARATOR = " / "
NAME_SEPARATOR = ","

# 본문은 처방 누적 80% 를 덮을 때까지, 최대 40개 (docs/dev/AI-6 §5).
COVERAGE = 0.8
MAX_EXERCISES = 40

# 프로젝트 전체의 저표본 기준 (docs/02 §5.3).
MIN_ROWS = 30

VOCABULARY_FILE = "exercise_vocabulary.csv"
CHUNKS_FILE = "prescription_chunks.csv"


def identity(name: str) -> str:
    """운동명의 동일성 키. **표기 차이까지만 합친다** (docs/dev/AI-6 §4 ②).

    NFKC 뒤 공백·가운뎃점을 지운다. 그 이상은 합치지 않는다 — 비슷해 보이는 이름
    (`스트레칭`·`스트레칭2`, `I`·`II`, `전방`·`후방`)은 대부분 다른 운동이다.
    """
    return re.sub(r"[\s·ㆍ]", "", unicodedata.normalize("NFKC", name))


def parse(text: str) -> dict[str, list[str]]:
    """처방문 하나를 단계별 운동명 목록으로. 모르는 단계 라벨은 버린다."""
    out: dict[str, list[str]] = {}
    for segment in str(text).split(PHASE_SEPARATOR):
        phase, sep, body = segment.partition(":")
        phase = phase.strip()
        if not sep or phase not in PHASES:
            continue
        names = [n.strip() for n in body.split(NAME_SEPARATOR)]
        out[phase] = [n for n in names if n]
    return out


@dataclass(frozen=True)
class Term:
    """어휘 한 줄. `name` 이 계약의 값이다."""

    name: str
    raw_forms: tuple[str, ...]
    count: int
    phases: tuple[str, ...]
    age_groups: tuple[str, ...]


def build_vocabulary(df: pd.DataFrame) -> list[Term]:
    """운동명의 닫힌 집합. 대표 표기는 가장 많이 쓰인 원문이다 (docs/dev/AI-6 §4 ②)."""
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    phases: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)

    for age_group, text in zip(df[M.AGE_GROUP_COL], df[M.PRESCRIPTION_COL], strict=True):
        for phase, names in parse(text).items():
            for raw in names:
                key = identity(raw)
                forms[key][raw] += 1
                phases[key].add(phase)
                groups[key].add(str(age_group))

    terms = [
        Term(
            name=counter.most_common(1)[0][0],
            raw_forms=tuple(sorted(counter)),
            count=sum(counter.values()),
            phases=tuple(p for p in PHASES if p in phases[key]),
            age_groups=tuple(sorted(groups[key])),
        )
        for key, counter in forms.items()
    ]
    return sorted(terms, key=lambda t: (-t.count, t.name))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    citation_label: str
    age_group: str
    age: int
    age_unit: str
    sex: str
    grade: str  # 등급 없는 층은 빈 문자열
    phase: str
    n: int
    exercise_names: tuple[str, ...]


def natural_key(age_group: str, age: int, sex: str, phase: str, grade: str = "") -> str:
    """**전부 구조로 정해진다.** 해시가 없다 (docs/dev/AI-6 §4 ①).

    원자료가 쌓이면 빈도가 바뀌어 내용 해시도 바뀌지만, 구조 키는 같은 칸이면 같다
    — 저장된 인용이 끊기지 않는다 (docs/04 §5). 등급 층은 등급이 키에 더 붙고, 등급
    없는 층의 키는 등급이 없던 때와 같다.
    """
    if grade:
        return f"{age_group}-{age}-{sex}-{grade}-{phase}"
    return f"{age_group}-{age}-{sex}-{phase}"


def citation_label(age_group: str, age: int, grade: str = "") -> str:
    """유아기는 개월이다 (docs/02 §2.4). 등급 층은 등급을 덧붙인다."""
    label = f"국민체력100 운동처방 · {age_group} {age}{age_unit_of(age_group)}"  # type: ignore[arg-type]
    return f"{label} · {grade}" if grade else label


def build_chunks(df: pd.DataFrame, vocabulary: list[Term]) -> tuple[list[Chunk], list[tuple]]:
    """등급 없는 층. 칸·단계마다 청크 하나. 표본 30 미만 칸은 만들지 않고 따로 돌려준다.

    등급을 모르는 사용자(측정값이 없거나 판정 불가)와 등급 칸이 비는 곳이 여기로
    떨어진다 (docs/04 §1.1).
    """
    return _aggregate(df, vocabulary, by_grade=False)


def build_grade_chunks(df: pd.DataFrame, vocabulary: list[Term]) -> tuple[list[Chunk], list[tuple]]:
    """등급 층. `(연령구간, 나이, 성별, 등급, 단계)` 마다 청크 하나.

    docs/04 §2.2·§3 이 처방 청크의 `grade` 를 메타데이터·필터로 규정했다. 같은 칸
    안에서도 등급에 따라 처방이 달라진다 (docs/dev/AI-6 §5). 등급이 비어 있는 행은
    이 층에서 빠지고 등급 없는 층에만 들어간다. 등급 열이 없으면 빈 층을 돌려준다.
    """
    if M.GRADE_COL not in df.columns:
        return [], []
    return _aggregate(df, vocabulary, by_grade=True)


def _aggregate(
    df: pd.DataFrame, vocabulary: list[Term], *, by_grade: bool
) -> tuple[list[Chunk], list[tuple]]:
    display = {identity(t.name): t.name for t in vocabulary}
    counts: dict[tuple[str, int, str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    rows: Counter[tuple[str, int, str, str]] = Counter()
    grades = list(fold_grades(df[M.GRADE_COL])) if by_grade else [""] * len(df)

    for age_group, age, sex, text, grade in zip(
        df[M.AGE_GROUP_COL],
        df[M.AGE_COL],
        df[M.SEX_COL],
        df[M.PRESCRIPTION_COL],
        grades,
        strict=True,
    ):
        if by_grade and grade not in GRADES:
            continue  # 등급 미기재 — 등급 없는 층에만 들어간다
        key = (str(age_group), int(age), str(sex), str(grade))
        rows[key] += 1
        for phase, names in parse(text).items():
            for raw in names:
                counts[key][phase][display[identity(raw)]] += 1

    chunks: list[Chunk] = []
    skipped: list[tuple] = []
    for key in sorted(counts):
        age_group, age, sex, grade = key
        n = rows[key]
        if n < MIN_ROWS:
            info = (age_group, age, sex, grade, n) if by_grade else (age_group, age, sex, n)
            skipped.append(info)
            continue
        for phase in PHASES:
            counter = counts[key].get(phase)
            if not counter:
                continue
            chosen = _cover(counter)
            chunks.append(
                Chunk(
                    chunk_id=f"{SOURCE}:{natural_key(age_group, age, sex, phase, grade)}",
                    text=_text(age_group, age, sex, grade, phase, n, chosen),
                    citation_label=citation_label(age_group, age, grade),
                    age_group=age_group,
                    age=age,
                    age_unit=age_unit_of(age_group),  # type: ignore[arg-type]
                    sex=sex,
                    grade=grade,
                    phase=phase,
                    n=n,
                    exercise_names=tuple(name for name, _ in chosen),
                )
            )
    return chunks, skipped


def _cover(counter: Counter[str]) -> list[tuple[str, float]]:
    """처방 누적 80% 를 덮을 때까지 싣는다. 비율은 그 단계 전체 처방 대비다."""
    total = sum(counter.values())
    chosen: list[tuple[str, float]] = []
    covered = 0
    for name, count in counter.most_common():
        if len(chosen) >= MAX_EXERCISES or covered / total >= COVERAGE:
            break
        chosen.append((name, count / total))
        covered += count
    return chosen


def _text(
    age_group: str,
    age: int,
    sex: str,
    grade: str,
    phase: str,
    n: int,
    chosen: list[tuple[str, float]],
) -> str:
    """임베딩 대상 본문. **원문을 센 것이지 요약한 것이 아니다** (docs/04 §2.1)."""
    who = f"{age_group} {age}{age_unit_of(age_group)} {'여자' if sex == 'F' else '남자'}"  # type: ignore[arg-type]
    if grade:
        who = f"{who} {grade}"
    listed = ", ".join(f"{name}({share:.0%})" for name, share in chosen)
    return f"{who} {n:,}명에게 처방된 {phase}: {listed}"


def vocabulary_frame(terms: list[Term]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exercise_name": t.name,
                "raw_forms": ";".join(t.raw_forms),
                "count": t.count,
                "phases": ";".join(t.phases),
                "age_groups": ";".join(t.age_groups),
                "fitness_factors": "",  # 비워 둔다 (docs/dev/AI-6 §4 ③)
            }
            for t in terms
        ]
    )


def chunks_frame(chunks: list[Chunk]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": c.chunk_id,
                "source": SOURCE,
                "text": c.text,
                "citation_label": c.citation_label,
                "citation_url": "",
                "age_group": c.age_group,
                "age": c.age,
                "age_unit": c.age_unit,
                "sex": c.sex,
                "grade": c.grade,
                "phase": c.phase,
                "n": c.n,
                "exercise_names": ";".join(c.exercise_names),
            }
            for c in chunks
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="처방 어휘와 처방 청크를 CSV로 낸다")
    ap.add_argument("--data-dir", required=True, help="원자료 월별 CSV 디렉터리")
    ap.add_argument("--release", default="data/release", help="어휘를 낼 곳 (커밋한다)")
    ap.add_argument("--interim", default="data/interim", help="청크를 낼 곳 (커밋하지 않는다)")
    args = ap.parse_args(argv)

    df = M.load_prescriptions(args.data_dir)
    vocabulary = build_vocabulary(df)
    chunks, skipped = build_chunks(df, vocabulary)
    graded, graded_skipped = build_grade_chunks(df, vocabulary)

    release, interim = Path(args.release), Path(args.interim)
    release.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    # utf-8-sig — 검수하는 사람이 엑셀로 연다 (docs/02 §4)
    csv = {"index": False, "encoding": "utf-8-sig"}
    vocabulary_frame(vocabulary).to_csv(release / VOCABULARY_FILE, **csv)
    chunks_frame(chunks + graded).to_csv(interim / CHUNKS_FILE, **csv)

    merged = sum(1 for t in vocabulary if len(t.raw_forms) > 1)
    print(f"처방 행 {len(df):,} · 어휘 {len(vocabulary):,} (표기 차이로 합친 것 {merged})")
    total = len(chunks) + len(graded)
    print(f"청크 {total:,} (등급 없는 층 {len(chunks):,} · 등급 층 {len(graded):,})")
    print(f"  → {interim / CHUNKS_FILE}")
    print(f"어휘 → {release / VOCABULARY_FILE}")
    if skipped:
        print(f"[알림] 표본 {MIN_ROWS} 미만이라 청크를 만들지 않은 칸 {len(skipped)}개")
        for ag, age, sex, n in skipped:
            print(f"  {ag} {age} {sex} · {n}명")
    if graded_skipped:
        dropped = sum(item[-1] for item in graded_skipped)
        print(
            f"[알림] 표본 {MIN_ROWS} 미만인 등급 칸 {len(graded_skipped)}개"
            f" ({dropped:,}행) — 등급 없는 층으로 떨어진다"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
