"""`POST /v1/coach/messages` 의 속 (docs/03 §6 · docs/04 §4 · dev/AI-10).

**LLM 이 찾은 청크로 문구를 쓰고, 발췌가 대체 경로다** (docs/01 §3.1 — 「`compose`
문장 생성 (LLM 1회)」 · 「실패 시 규칙 편성으로 강등」). `COACH_LLM` 이 꺼져 있으면
발췌만 나가고 외부 호출이 0이다.

**LLM 이 쓴 답도 검사를 통과해야 나간다** (docs/04 §4.2) — 인용이 없거나 본문의
`[n]` 이 인용 범위를 벗어나면 `no_citation_generated` 다. 「모든 응답이 이 검사를
통과한다. 우회 경로를 두지 않는다.」

거부가 세 갈래로 먼저 온다 (docs/04 §4.1).

| 조건 | `refusal_reason` | 검색하나 |
|---|---|---|
| 부상·통증·질환·약물 질의 | `medical_query` | **안 한다** |
| 연령 필터 뒤 0건 | `age_filter_empty` | 한다 |
| 임계값 넘는 청크 0건 | `no_relevant_source` | 한다 |
| 본문 `[n]` 이 인용 밖 | `no_citation_generated` | 한다 |

**거부는 200 이다.** 오류가 아니고 호출자는 재시도하지 않는다 (docs/04 §4.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..api.coach_schemas import Citation, CoachMessageResponse
from ..common.copy import contains_forbidden
from ..rag.medical import MEDICAL_COPY, NO_SOURCE_COPY, matched
from ..rag.search import Corpus, Hit
from .llm import NO_BASIS, AnswerWriter
from .verify import cite_numbers

# 발췌 한 조각의 길이. 청크는 50~400토큰이라 통째로 내면 화면을 덮는다.
EXCERPT_CHARS = 220

# 한 답변에 넣는 인용 수. `search.MAX_CONTEXT` 가 상한이고 그보다 좁게 쓴다 —
# 근거 다섯 개를 나열하는 것은 답이 아니라 검색 결과다.
MAX_CITED = 3


@dataclass(frozen=True)
class Answered:
    response: CoachMessageResponse
    # 로그와 평가용. 응답에 싣지 않는다 (질문·답변은 로그 금지 필드다).
    hit_count: int = 0
    filtered_out: dict[str, int] | None = None
    medical_terms: tuple[str, ...] = ()
    # 문구를 무엇이 썼나 — `발췌` · `claude:...` · `발췌(강등 · 이유)`. 로그에 남는다
    source: str = "발췌"


def _refuse(reason: str, copy: str, **extra: object) -> Answered:
    """거부. **`source` 를 함께 남긴다** — 어느 갈래가 거부했는지 로그에서 알 수 있게.

    검색이 못 찾은 것과 모델이 못 쓴 것은 다른 문제다. 구분되지 않으면 임계값을
    고쳐야 하는지 프롬프트를 고쳐야 하는지 알 수 없다 (2026-09-16 실측에서 막혔다).
    """
    extra.setdefault("source", f"거부:{reason}")
    return Answered(
        response=CoachMessageResponse(
            answer=copy,
            citations=[],
            refused=True,
            refusal_reason=reason,  # type: ignore[arg-type]
        ),
        **extra,  # type: ignore[arg-type]
    )


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """청크에서 한 조각. **문장을 끊어 뜻을 바꾸지 않는다** — 마지막 온전한 구분점까지만.

    자료의 글자를 고치지 않는다 (docs/02 §4 — 원본 문자열 보존).
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    window = flat[:limit]
    for mark in (". ", "다 ", "요 ", "· ", ", "):
        cut = window.rfind(mark)
        if cut > limit // 2:
            return window[: cut + len(mark)].strip()
    return window.rstrip() + "…"


def compose(question: str, hits: tuple[Hit, ...]) -> tuple[str, list[Citation]]:
    """발췌를 이어 답을 만들고 `[n]` 을 박는다.

    **인용은 본문에 쓴 것만 싣는다.** 쓰지 않은 근거를 목록에 넣으면 `[n]` 과 인용이
    어긋나고, `verify` 가 그것을 걸러낸다.
    """
    used = hits[:MAX_CITED]
    citations = [
        Citation(
            index=number,
            label=hit.citation_label,
            chunk_id=hit.chunk_id,
            url=hit.citation_url or None,
        )
        for number, hit in enumerate(used, start=1)
    ]
    pieces = [f"{excerpt(hit.text)} [{number}]" for number, hit in enumerate(used, start=1)]
    return " ".join(pieces), citations


def _empty_reason(filtered_out: dict[str, int]) -> str:
    """0건이 **왜** 0건인가 (docs/04 §4.1).

    `below_threshold` 가 있으면 그 연령대 후보는 있었고 하나도 임계값을 넘지 못한
    것이다 → `no_relevant_source`. 후보가 아예 없었고 연령으로 빠진 것만 있으면
    → `age_filter_empty`.

    **연령으로 빠진 수만 보고 가르면 안 된다.** 무관한 질의도 top-k 에는 이웃이
    잡히고 그중 다른 연령대가 섞여 `age_group` 이 0보다 커진다 — 「주식 투자
    어떻게 시작해요」가 `age_filter_empty` 로 거부되던 이유다 (2026-09-16 실측).
    """
    if filtered_out.get("below_threshold", 0):
        return "no_relevant_source"
    return "age_filter_empty" if filtered_out.get("age_group", 0) else "no_relevant_source"


def cited_within(body: str, citations: list[Citation]) -> bool:
    """본문의 `[n]` 이 전부 인용 범위 안이고, 적어도 하나는 있나 (docs/04 §4.2).

    **답을 고쳐서 통과시키지 않는다.** 범위 밖 번호를 지우면 그 문장의 근거가
    사라진 채 남는다 — 통째로 강등하거나 거부하는 것이 정직하다.
    """
    used = set(cite_numbers(body))
    return bool(used) and used <= {c.index for c in citations}


def answer(
    question: str,
    age_group: str,
    corpus: Corpus,
    embed_query: object,
    threshold: float,
    writer: AnswerWriter | None = None,
) -> Answered:
    """`embed_query` 는 질의 하나를 정규화 벡터로 바꾸는 것이다.

    `writer` 가 있으면 LLM 이 문구를 쓴다. **의료 질의는 임베더도 LLM 도 부르지
    않는다** — 시험이 그것을 단정으로 잠근다.
    """
    if terms := matched(question):
        return _refuse("medical_query", MEDICAL_COPY, medical_terms=tuple(terms))

    vector = embed_query(question)  # type: ignore[operator]
    result = corpus.search(vector, [age_group], threshold)

    if not result.hits:
        return _refuse(
            _empty_reason(result.filtered_out), NO_SOURCE_COPY, filtered_out=result.filtered_out
        )

    excerpted, citations = compose(question, result.hits)
    if not citations:
        return _refuse("no_citation_generated", NO_SOURCE_COPY, filtered_out=result.filtered_out)

    body, source = excerpted, "발췌"
    if writer is not None:
        try:
            written = writer.write(question, result.hits[:MAX_CITED])
        except Exception as failure:  # 강등이지 거부가 아니다 (docs/01 §3.1)
            body, source = excerpted, f"발췌(강등 · {type(failure).__name__})"
        else:
            if written.strip() == NO_BASIS:
                # **거부가 아니라 강등이다.** 관련성은 잰 임계값이 정한다 (docs/04 §3) —
                # 모델에게 그 판단을 맡기면 「무엇을 근거로 삼을지」를 LLM 이 정하는
                # 것이고 AGENTS.md §7 에 어긋난다. 실측: 검색이 인용 3개를 찾은 질의
                # 5개 중 2개에 모델이 `근거없음` 을 냈다 — 답할 수 있는 물음이었다.
                body, source = excerpted, "발췌(강등 · 모델이 근거없음)"
            elif contains_forbidden(written) or not cited_within(written, citations):
                body, source = excerpted, "발췌(강등 · 인용·어휘 검사)"
            else:
                body, source = written, writer.version()

    if not cited_within(body, citations):
        # 발췌 경로가 이것을 깨뜨릴 수는 없지만, 규약이 「우회 경로를 두지 않는다」다
        return _refuse("no_citation_generated", NO_SOURCE_COPY, filtered_out=result.filtered_out)

    return Answered(
        response=CoachMessageResponse(
            answer=body, citations=citations, refused=False, refusal_reason=None
        ),
        hit_count=len(result.hits),
        filtered_out=result.filtered_out,
        source=source,
    )
