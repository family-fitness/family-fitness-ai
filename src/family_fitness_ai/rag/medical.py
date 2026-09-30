"""의료 질의 가리기.

부상·통증·질환·약물 질의는 **검색조차 하지 않고** 거부한다. 우리가 가진 자료는
국민체력100 측정·처방이지 진료 자료가 아니다. 애매하면 거부 쪽으로 기운다 —
답을 못 주는 것보다 잘못 주는 것이 나쁘다.

낱말이 아니라 조각으로 찾는다. 한국어는 조사와 활용이 붙어 낱말 경계가 흐리다.
「아프다」 하나가 아파·아픈·아픔·아팠 으로 갈라지니 어간을 묶어 적는다.

맞고 틀린 정도는 tests/fixtures/medical_queries.csv 로 잰다:
    python -m family_fitness_ai.rag.medical
"""

from __future__ import annotations

import re

_PATTERNS = (
    # 부상·통증
    r"다[치쳐쳤친칠]",
    r"부상",
    r"삐[었어끗]",
    r"골절|인대|연골|탈구|염좌",
    r"통증",
    r"아[프파픈픔팠픕]",
    r"쑤시|결[리려]|저[리려]|멍이",
    # 질환·진료
    r"질환|질병|증상|진단|치료|수술|재활|물리치료",
    r"디스크|측만|관절염|천식|당뇨|고혈압|심장|빈혈|골다공증",
    r"발작|경련|어지[럼러]|구토|설사|알레르기|아토피",
    r"성장통|거북목|평발|비만",
    r"병원|의사|한의원|정형외과|소아과|재활의학",
    # 약물
    r"약[을이은 ]|복용|진통제|해열제|항생제|주사|처방전",
    # 몸무게를 겨냥한 질문 — 아이에게 답할 자료가 우리에게 없다
    r"살\s*[빼뺄뺴]|감량|다이어트|체중\s*줄|몇\s*킬로",
)

_PATTERN = re.compile("|".join(_PATTERNS))


def is_medical(question: str) -> bool:
    return bool(_PATTERN.search(question))


def matched_terms(question: str) -> list[str]:
    """어느 말에 걸렸는지. 로그와 시험에서 본다."""
    return sorted({match.group(0) for match in _PATTERN.finditer(question)})


def _report() -> None:
    """고정 질의 묶음으로 오탐·누락을 센다."""
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "medical_queries.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    wrong = []
    for row in rows:
        expected = row["medical"] == "1"
        actual = is_medical(row["question"])
        if expected != actual:
            wrong.append((row["question"], expected, actual))

    false_positive = sum(1 for _, expected, _ in wrong if not expected)
    false_negative = sum(1 for _, expected, _ in wrong if expected)
    print(f"질의 {len(rows)}건 · 오탐 {false_positive} · 누락 {false_negative}")
    for question, expected, actual in wrong:
        label = "오탐" if not expected else "누락"
        print(f"  {label}: {question} (기대 {expected} / 실제 {actual})")


if __name__ == "__main__":
    _report()
