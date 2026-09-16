"""의료 질의를 가려낸다 (docs/04 §4.1).

부상·통증·질환·약물 질의는 **검색조차 하지 않고** 거부한다 (docs/03 §6). 없는
근거로 답을 만들 기회를 주지 않는 것이 프롬프트로 당부하는 것보다 확실하다.

**LLM 에게 판정을 맡기지 않는다** (AGENTS.md §7). 낱말 목록으로 가린다 — 무엇이
걸리고 무엇이 안 걸리는지 사람이 읽을 수 있어야 하고, 걸린 이유가 기록에 남아야 한다.

**부위 낱말만으로는 걸리지 않는다.** `무릎`·`허리` 는 운동 질의에도 흔히 나온다
(`무릎 굽혀 가슴 닿기` 는 처방 어휘다). 증상·질환·처치 낱말이 있어야 걸린다.

재는 방법 — 오탐은 기존 질의 세트로, 누락은 의료 질의 세트로 센다:
    python -m family_fitness_ai.rag.medical --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..common.settings import RELEASE_DIR

MEDICAL_TERMS_FILE = "medical_terms.csv"
MEDICAL_QUERIES_FILE = "medical_queries.csv"

# docs/04 §4.3 — 거부 문구는 글자 그대로 이것이다.
MEDICAL_COPY = "운동 중 통증이 있다면 전문가와 상담하세요."
NO_SOURCE_COPY = "해당 내용은 국민체력100 자료에서 찾지 못했습니다."

# 갈래 — `증상`·`질환`·`처치`·`약물`. 부위는 넣지 않는다 (위 규약).
#
# **넣은 근거**: docs/04 §4.1 이 「부상·통증·질환·약물」 넷을 지목했다. 그 넷을
# 한국어로 실제 물을 때 쓰는 낱말을 담았고, 처방 어휘 641개와 겹치는 것은 뺐다.
TERMS: dict[str, tuple[str, ...]] = {
    "증상": (
        "아파",
        "아프",
        "아픈",
        "아픔",
        "통증",
        "쑤신",
        "쑤셔",
        "저림",
        "저려",
        "저리",
        "붓기",
        "부었",
        # `부어` 만 두면 「한글 공부 어떻게」가 걸린다 (실측 오탐 q37)
        "부어올",
        "시큰",
        "결려",
        "결림",
        # `삐` 한 자만 두면 운동 이름에도 걸린다
        "삐었",
        "삐어",
        "쥐가",
        # `통증` 으로 잡히지 않는다 (실측 누락)
        "성장통",
    ),
    "부상": (
        "부상",
        "다쳤",
        "다치",
        "다쳐",
        "골절",
        "인대",
        "파열",
        "염좌",
        "탈구",
        "디스크",
        "수술",
        "재활",
        "깁스",
    ),
    "질환": (
        "질환",
        "질병",
        "염증",
        "관절염",
        "건염",
        "근막염",
        "협착",
        "당뇨",
        "고혈압",
        "심장병",
        "천식",
        "골다공증",
        # `암` 한 자는 넣지 않는다 — 처방 어휘 「암 워킹」에 걸린다
        "진단",
        "증후군",
    ),
    "처치": (
        "병원",
        "의사",
        "한의원",
        "물리치료",
        "도수치료",
        "주사",
        "정형외과",
        "재활치료",
        "치료",
        "처치",
    ),
    "약물": (
        "약물",
        # `약을` 은 「계약을」에 걸린다 — `복용`·`진통제` 로 잡는다
        "복용",
        "진통제",
        "소염제",
        "파스",
        "영양제",
        "보충제",
        "스테로이드",
        "처방약",
    ),
}


def load_terms(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """CSV 가 있으면 그것이 정본이다. 없으면 코드의 `TERMS` 를 쓴다.

    두 벌을 두는 것이 아니다 — CSV 는 `TERMS` 에서 만들어 내는 산출물이고, 검수하는
    사람이 엑셀로 열어 보는 자리다. 서비스는 CSV 를 읽는다.
    """
    target = path or (RELEASE_DIR / MEDICAL_TERMS_FILE)
    if not target.exists():
        return TERMS
    frame = pd.read_csv(target, encoding="utf-8-sig")
    out: dict[str, list[str]] = {}
    for row in frame.itertuples(index=False):
        out.setdefault(str(row.category), []).append(str(row.term))
    return {k: tuple(v) for k, v in out.items()}


def terms_frame(terms: dict[str, tuple[str, ...]] | None = None) -> pd.DataFrame:
    source = terms or TERMS
    return pd.DataFrame(
        [
            {"category": category, "term": term, "basis": "docs/04 §4.1"}
            for category, words in source.items()
            for term in words
        ]
    )


def matched(question: str, terms: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """걸린 낱말 전부. 빈 목록이면 의료 질의가 아니다.

    **무엇이 걸렸는지 돌려준다** — 걸린 이유가 로그에 남아야 오탐을 고칠 수 있다.
    """
    table = terms if terms is not None else load_terms()
    # **공백을 지우지 않는다.** 지우면 낱말 경계를 넘어 걸린다 — 「한글 공부 어떻게」가
    # `부어` 에 걸리는 오탐을 그렇게 만들었다 (2026-09-16 실측).
    return [term for words in table.values() for term in words if term in question]


def is_medical(question: str, terms: dict[str, tuple[str, ...]] | None = None) -> bool:
    return bool(matched(question, terms))


# ── 재기 ────────────────────────────────────────────────────────────

# 누락을 재는 질의. **`sim_queries.csv` 에 넣지 않는다** — 저쪽은 정답 청크가 있는
# 세트이고 `threshold.scores()` 의 분모다. 정답 없는 행을 더하면 `false_positive` 가
# 늘어 잰 정밀도가 재현되지 않는다.
MEDICAL_QUERIES: tuple[str, ...] = (
    "무릎이 아픈데 어떤 운동을 해야 해요?",
    "허리 디스크가 있어도 윗몸일으키기 해도 되나요?",
    "발목을 삐었는데 언제부터 뛸 수 있어요?",
    "아이가 운동하다 다쳤어요 어떻게 해야 하나요?",
    "어깨에 염증이 있다는데 스트레칭 괜찮을까요?",
    "당뇨가 있는데 운동 강도를 어떻게 잡나요?",
    "고혈압 약을 복용 중인데 근력운동 해도 됩니까",
    "무릎 수술 후 재활 운동 알려주세요",
    "관절염에 좋은 운동이 뭐예요?",
    "운동 전에 진통제를 먹어도 되나요",
    "종아리에 쥐가 자주 나는데 병원 가야 하나요",
    "아이 성장통이 있으면 운동을 쉬어야 하나요?",
)


def check(release: Path = RELEASE_DIR) -> tuple[int, int, list[str], list[str]]:
    """(누락, 오탐, 놓친 의료 질의, 잘못 걸린 운동 질의).

    오탐은 **기존 질의 세트 48문항**으로 센다 — 운동 질의가 의료로 걸리면 답변이
    나가지 않는다. 누락은 위 12문항으로 센다.
    """
    terms = load_terms(release / MEDICAL_TERMS_FILE)

    missed = [q for q in MEDICAL_QUERIES if not is_medical(q, terms)]

    queries = pd.read_csv(release / "sim_queries.csv", encoding="utf-8-sig")
    wrong = [
        f"{row.query_id} {row.query} ← {matched(str(row.query), terms)}"
        for row in queries.itertuples(index=False)
        if is_medical(str(row.query), terms)
    ]
    return len(missed), len(wrong), missed, wrong


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="의료 질의 목록을 내고 오탐·누락을 센다")
    ap.add_argument("--release", default=str(RELEASE_DIR), help="목록을 낼 곳 (커밋한다)")
    ap.add_argument("--check", action="store_true", help="내지 않고 재기만 한다")
    args = ap.parse_args(argv)

    release = Path(args.release)
    if not args.check:
        release.mkdir(parents=True, exist_ok=True)
        # utf-8-sig — 검수하는 사람이 엑셀로 연다 (docs/02 §4 ③)
        terms_frame().to_csv(release / MEDICAL_TERMS_FILE, index=False, encoding="utf-8-sig")
        pd.DataFrame({"query": MEDICAL_QUERIES}).to_csv(
            release / MEDICAL_QUERIES_FILE, index=False, encoding="utf-8-sig"
        )
        print(f"낱말 {len(terms_frame())}개 → {release / MEDICAL_TERMS_FILE}")
        print(f"질의 {len(MEDICAL_QUERIES)}개 → {release / MEDICAL_QUERIES_FILE}")

    missed, wrong, missed_list, wrong_list = check(release)
    print(f"누락 {missed}/{len(MEDICAL_QUERIES)} · 오탐 {wrong}/48")
    for item in missed_list:
        print(f"  [누락] {item}")
    for item in wrong_list:
        print(f"  [오탐] {item}")
    return 1 if (missed or wrong) else 0


if __name__ == "__main__":
    raise SystemExit(main())
