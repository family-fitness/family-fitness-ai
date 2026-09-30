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
from family_fitness_ai.rag.index import Chunk
from family_fitness_ai.rag.medical import is_medical
from family_fitness_ai.rag.search import search
from family_fitness_ai.video.vocabulary import exercises_in

MAX_PASSAGES = 3


def _refuse(reason: str) -> dict[str, object]:
    return {"answer": "", "citations": [], "refused": True, "refusal_reason": reason}


def _extract(chunk_text: str, source: str, index: int) -> str:
    """자료에서 그대로 뽑아 쓴 한 문장. 새로 지어내는 말이 없다."""
    if source == "prescription":
        names = [name for name, _ in exercises_in(chunk_text)][:3]
        if names:
            # 누가 받은 처방인지를 자료 문장 그대로 앞에 둔다. 「유연성 1등급」만 떼어
            # 쓰면 유연성을 기르는 운동 목록처럼 읽힌다.
            who = chunk_text.split(":", 1)[0].strip()
            return f"{who}에는 {', '.join(names)} 등이 있습니다 [{index}]."
    if source == "criteria":
        return f"{chunk_text} [{index}]."
    head = chunk_text.split(" · 나오는 운동:")[0]
    return f"{head} [{index}]."


def _passage(chunk: Chunk) -> str:
    """LLM 에 넘기는 근거 한 문단.

    처방 청크는 「그 등급인 사람들이 받은 본운동 처방 전체」다. 그대로 넘기면 LLM 이
    「유연성을 기르기 위해 처방된 운동」처럼 자료에 없는 효과를 붙였다. 무엇을 적은
    목록인지 문단 끝에 밝혀 둔다.
    """
    if chunk.source != "prescription":
        return chunk.text
    note = "이 목록은 이 사람들이 받은 본운동 처방 전체다."
    factor = "·".join(chunk.factors)
    if factor:
        note += f" {factor}만을 위한 운동 목록이 아니다."
    return f"{chunk.text} ({note})"


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
    passages = [(i + 1, _passage(hit.chunk)) for i, hit in enumerate(result.hits)]

    written = copywriter.write_answer(question, passages)
    if written is None:
        written = " ".join(
            _extract(hit.chunk.text, hit.chunk.source, i + 1)
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
