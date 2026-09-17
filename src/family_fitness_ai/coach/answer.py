"""운동·체력 질문에 자료를 찾아 답한다.

자료에 없으면 답하지 않는다. 부상·통증·질환·약물 질의는 **검색조차 하지 않고**
거부한다 — 우리가 가진 것은 국민체력100 측정·처방이지 진료 자료가 아니다.

LLM 이 꺼져 있어도 답은 나간다. 자료에서 그대로 뽑아 쓴 문장이라 인용이 늘
붙는다. LLM 은 그 문장을 읽기 좋게 바꿔 줄 뿐이고, 켜져 있어도 자료 밖으로
나가면 버린다.
"""

from __future__ import annotations

from family_fitness_ai.coach import llm as copywriter
from family_fitness_ai.coach import verify
from family_fitness_ai.rag.medical import is_medical
from family_fitness_ai.rag.search import search
from family_fitness_ai.video.vocabulary import exercises_in

MAX_PASSAGES = 3


def _refuse(reason: str) -> dict[str, object]:
    return {"answer": "", "citations": [], "refused": True, "refusal_reason": reason}


def _extract(chunk_text: str, source: str, label: str, index: int) -> str:
    """자료에서 그대로 뽑아 쓴 한 문장. 새로 지어내는 말이 없다."""
    if source == "prescription":
        names = [name for name, _ in exercises_in(chunk_text)][:3]
        if names:
            return f"{label}에서 자주 제시되는 본운동은 {', '.join(names)} 입니다 [{index}]."
    if source == "criteria":
        return f"{chunk_text} [{index}]."
    head = chunk_text.split(" · 나오는 운동:")[0]
    return f"{head} [{index}]."


def answer(question: str, age_group: str = "", profile_ref: str = "") -> dict[str, object]:
    if is_medical(question):
        return _refuse("medical_query")

    result = search(
        question,
        k=MAX_PASSAGES,
        sources=("prescription", "criteria", "video"),
        age_group=age_group or None,
    )
    if not result.hits:
        if age_group and result.filtered_out.get("age_group"):
            return _refuse("age_filter_empty")
        return _refuse("no_relevant_source")

    citations = [hit.chunk.citation(i + 1) for i, hit in enumerate(result.hits)]
    passages = [(i + 1, hit.chunk.text) for i, hit in enumerate(result.hits)]

    written = copywriter.write_answer(question, passages)
    if written is None:
        written = " ".join(
            _extract(hit.chunk.text, hit.chunk.source, hit.chunk.citation_label, i + 1)
            for i, hit in enumerate(result.hits[:2])
        )

    problems = verify.check_answer(written, citations)
    if problems:
        return _refuse("no_citation_generated")

    return {
        "answer": written,
        "citations": citations,
        "refused": False,
        "refusal_reason": None,
    }
