"""`POST /v1/coach/messages` (docs/03 §6 · docs/04 §4 · dev/AI-10).

**가짜 임베더와 작은 색인으로 돈다** — 시험이 외부를 부르지 않는다. LLM 도 가짜를
끼워 넣는다 (`AnswerWriter` 프로토콜).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from family_fitness_ai.api.coach_schemas import Citation
from family_fitness_ai.coach.answer import (
    MAX_CITED,
    _empty_reason,
    answer,
    cited_within,
    compose,
    excerpt,
)
from family_fitness_ai.coach.llm import NO_BASIS, WriterFailed, build_prompt
from family_fitness_ai.rag.medical import MEDICAL_COPY, NO_SOURCE_COPY
from family_fitness_ai.rag.search import Hit, Result

THRESHOLD = 0.5


def _hit(n: int, *, age: str = "유아기", text: str | None = None) -> Hit:
    return Hit(
        chunk_id=f"prescription:칸{n}",
        source="prescription",
        text=text or f"유아기 {48 + n}개월 처방 준비운동: 흔들어 체조, 거북이 스트레칭",
        citation_label=f"국민체력100 운동처방 · 유아기 {48 + n}개월",
        citation_url="",
        age_group=age,
        score=0.9 - n * 0.01,
    )


class _Corpus:
    """`search.Corpus` 를 대신한다. 무엇을 돌려줄지 시험이 정한다."""

    def __init__(self, hits: tuple[Hit, ...], filtered_out: dict[str, int] | None = None):
        self._hits = hits
        self._filtered = filtered_out or {"age_group": 0, "below_threshold": 0}
        self.searched = 0

    def search(self, vector, age_groups, threshold, **kw) -> Result:  # noqa: ANN001
        self.searched += 1
        return Result(self._hits, self._filtered)


class _Embedder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> np.ndarray:
        self.calls += 1
        return np.zeros(4, dtype="float32")


class _Writer:
    """가짜 LLM. 쓸 문구를 시험이 정한다."""

    def __init__(self, text: str | None = None, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0

    def version(self) -> str:
        return "claude:가짜"

    def write(self, question: str, hits: Sequence[Hit]) -> str:
        self.calls += 1
        if self.fail:
            raise WriterFailed("시험용 실패")
        return self.text or "흔들어 체조와 거북이 스트레칭이 함께 제시됩니다 [1]."


# ── 거부 (docs/04 §4.1) ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "무릎이 아픈데 어떤 운동을 해야 해요?",
        "허리 디스크가 있어도 윗몸일으키기 해도 되나요?",
        "관절염에 좋은 운동이 뭐예요?",
        "운동 전에 진통제를 먹어도 되나요",
    ],
)
def test_의료_질의는_검색도_llm도_부르지_않는다(question: str) -> None:
    """「없는 근거로 답을 만들 기회를 주지 않는다」 (docs/04 §4.1)."""
    corpus, embedder, writer = _Corpus((_hit(1),)), _Embedder(), _Writer()
    result = answer(question, "유소년", corpus, embedder, THRESHOLD, writer=writer)  # type: ignore[arg-type]
    assert result.response.refused is True
    assert result.response.refusal_reason == "medical_query"
    assert result.response.answer == MEDICAL_COPY
    assert result.response.citations == []
    assert embedder.calls == 0, "의료 질의에 임베더를 불렀다"
    assert corpus.searched == 0, "의료 질의에 검색을 했다"
    assert writer.calls == 0, "의료 질의에 LLM 을 불렀다"


def test_임계값_미달은_no_relevant_source() -> None:
    corpus = _Corpus((), {"age_group": 0, "below_threshold": 7})
    result = answer("운동 추천", "유아기", corpus, _Embedder(), THRESHOLD)  # type: ignore[arg-type]
    assert result.response.refusal_reason == "no_relevant_source"
    assert result.response.answer == NO_SOURCE_COPY


def test_연령으로만_빠졌으면_age_filter_empty() -> None:
    corpus = _Corpus((), {"age_group": 5, "below_threshold": 0})
    result = answer("운동 추천", "유소년", corpus, _Embedder(), THRESHOLD)  # type: ignore[arg-type]
    assert result.response.refusal_reason == "age_filter_empty"


def test_무관한_질의는_연령이_섞여도_no_relevant_source() -> None:
    """무관한 질의도 top-k 에는 이웃이 잡히고 그중 다른 연령대가 섞인다 —
    연령으로 빠진 수만 보고 가르면 「주식 투자」가 `age_filter_empty` 가 된다."""
    corpus = _Corpus((), {"age_group": 9, "below_threshold": 1})
    result = answer("주식 투자 어떻게 시작해요", "성인", corpus, _Embedder(), THRESHOLD)  # type: ignore[arg-type]
    assert result.response.refusal_reason == "no_relevant_source"


@pytest.mark.parametrize(
    ("filtered", "expected"),
    [
        ({"below_threshold": 3, "age_group": 0}, "no_relevant_source"),
        ({"below_threshold": 3, "age_group": 9}, "no_relevant_source"),
        ({"below_threshold": 0, "age_group": 9}, "age_filter_empty"),
        ({"below_threshold": 0, "age_group": 0}, "no_relevant_source"),
    ],
)
def test_빈_결과의_사유(filtered: dict[str, int], expected: str) -> None:
    assert _empty_reason(filtered) == expected


# ── 발췌 경로 (LLM 꺼짐) ────────────────────────────────────────────


def test_발췌는_인용을_박고_범위_안이다() -> None:
    corpus = _Corpus(tuple(_hit(n) for n in range(1, 5)))
    result = answer("유아기 준비운동 알려줘", "유아기", corpus, _Embedder(), THRESHOLD)  # type: ignore[arg-type]
    body = result.response.answer
    assert result.response.refused is False
    assert result.response.citations
    assert len(result.response.citations) <= MAX_CITED
    assert cited_within(body, list(result.response.citations))
    assert result.source == "발췌"


def test_인용은_본문에_쓴_것만_싣는다() -> None:
    """쓰지 않은 근거를 목록에 넣으면 `[n]` 과 인용이 어긋난다."""
    hits = tuple(_hit(n) for n in range(1, 6))
    body, citations = compose("물음", hits)
    assert len(citations) == MAX_CITED
    assert {c.index for c in citations} == set(range(1, MAX_CITED + 1))
    assert cited_within(body, citations)


def test_citation_label_이_비지_않는다() -> None:
    """`AiWire.CitationBody.label` 은 기본값이 없다 — 비면 역직렬화가 터진다."""
    corpus = _Corpus(tuple(_hit(n) for n in range(1, 4)))
    result = answer("물음", "유아기", corpus, _Embedder(), THRESHOLD)  # type: ignore[arg-type]
    assert all(c.label.strip() for c in result.response.citations)
    assert all(c.chunk_id.strip() for c in result.response.citations)


def test_발췌는_문장을_끊어_뜻을_바꾸지_않는다() -> None:
    long = "앞 문장이다. " * 40
    cut = excerpt(long, limit=50)
    assert len(cut) <= 52
    assert cut.endswith(("다.", "다", "…"))


def test_짧은_청크는_그대로다() -> None:
    assert excerpt("짧다") == "짧다"


# ── LLM 경로 ────────────────────────────────────────────────────────


def test_llm_이_쓴_문구가_나간다() -> None:
    corpus, writer = _Corpus(tuple(_hit(n) for n in range(1, 4))), _Writer()
    result = answer("물음", "유아기", corpus, _Embedder(), THRESHOLD, writer=writer)  # type: ignore[arg-type]
    assert writer.calls == 1
    assert result.response.answer.endswith("[1].")
    assert result.source == "claude:가짜"


def test_llm_이_실패하면_발췌로_강등한다() -> None:
    """**거부가 아니라 강등이다** (docs/01 §3.1). 그리고 강등이 로그에 남는다."""
    corpus, writer = _Corpus(tuple(_hit(n) for n in range(1, 4))), _Writer(fail=True)
    result = answer("물음", "유아기", corpus, _Embedder(), THRESHOLD, writer=writer)  # type: ignore[arg-type]
    assert result.response.refused is False
    assert result.response.citations
    assert "강등" in result.source and "WriterFailed" in result.source


def test_범위_밖_인용을_쓰면_강등한다() -> None:
    """답을 고쳐서 통과시키지 않는다 — 통째로 발췌로 바꾼다."""
    corpus = _Corpus(tuple(_hit(n) for n in range(1, 4)))
    writer = _Writer("근거에 없는 말입니다 [9].")
    result = answer("물음", "유아기", corpus, _Embedder(), THRESHOLD, writer=writer)  # type: ignore[arg-type]
    assert result.response.refused is False
    assert "[9]" not in result.response.answer
    assert "강등" in result.source


def test_인용이_아예_없으면_강등한다() -> None:
    corpus = _Corpus(tuple(_hit(n) for n in range(1, 4)))
    writer = _Writer("근거 번호를 안 붙인 답입니다.")
    result = answer("물음", "유아기", corpus, _Embedder(), THRESHOLD, writer=writer)  # type: ignore[arg-type]
    assert "강등" in result.source
    assert cited_within(result.response.answer, list(result.response.citations))


def test_금지_어휘를_쓰면_강등한다() -> None:
    corpus = _Corpus(tuple(_hit(n) for n in range(1, 4)))
    writer = _Writer("유연성이 부족합니다 [1].")
    result = answer("물음", "유아기", corpus, _Embedder(), THRESHOLD, writer=writer)  # type: ignore[arg-type]
    assert "부족" not in result.response.answer
    assert "강등" in result.source


def test_근거없음이면_거부가_아니라_강등이다() -> None:
    """**관련성은 잰 임계값이 정한다** (docs/04 §3).

    모델에게 그 판단을 맡기면 「무엇을 근거로 삼을지」를 LLM 이 정하는 것이고
    AGENTS.md §7 에 어긋난다. 실측: 검색이 인용 3개를 찾은 질의 5개 중 3개에 모델이
    `근거없음` 을 냈다 — 답할 수 있는 물음이었고, 거부로 두었더니 2개가 버려졌다.
    """
    corpus = _Corpus(tuple(_hit(n) for n in range(1, 4)))
    result = answer("물음", "유아기", corpus, _Embedder(), THRESHOLD, writer=_Writer(NO_BASIS))  # type: ignore[arg-type]
    assert result.response.refused is False
    assert result.response.citations, "찾은 근거를 버리지 않는다"
    assert result.response.answer != NO_SOURCE_COPY
    assert "근거없음" in result.source


def test_거부에도_어느_갈래인지_남는다() -> None:
    """검색이 못 찾은 것과 모델이 못 쓴 것은 다른 문제다 — 구분되지 않으면
    임계값을 고쳐야 하는지 프롬프트를 고쳐야 하는지 알 수 없다."""
    medical = answer("무릎이 아파요", "유소년", _Corpus((_hit(1),)), _Embedder(), THRESHOLD)  # type: ignore[arg-type]
    assert medical.source == "거부:medical_query"

    empty = _Corpus((), {"age_group": 0, "below_threshold": 4})
    none_found = answer("물음", "유아기", empty, _Embedder(), THRESHOLD)  # type: ignore[arg-type]
    assert none_found.source == "거부:no_relevant_source"


def test_키는_repr_에_실리지_않는다() -> None:
    """dataclass 의 기본 `repr` 이 값을 그대로 찍어 단정문 실패 메시지에 키가 실렸다."""
    from family_fitness_ai.coach.llm import ClaudeWriter, GeminiWriter

    secret = "이건비밀키다"
    for writer in (ClaudeWriter(api_key=secret), GeminiWriter(api_key=secret)):
        assert secret not in repr(writer)
        assert writer.api_key == secret, "값은 살아 있어야 한다"


def test_프롬프트에_근거_번호가_인용과_같다() -> None:
    hits = [_hit(1), _hit(2)]
    prompt = build_prompt("물음", hits)
    assert "[1]" in prompt and "[2]" in prompt
    assert "[3]" not in prompt
    for hit in hits:
        assert hit.text in prompt
        assert hit.citation_label in prompt
    assert "물음" in prompt


def test_cited_within() -> None:
    cites = [
        Citation(index=1, label="ㄱ", chunk_id="a"),
        Citation(index=2, label="ㄴ", chunk_id="b"),
    ]
    assert cited_within("가 [1] 나 [2]", cites) is True
    assert cited_within("가 [1]", cites) is True
    assert cited_within("가 [3]", cites) is False
    assert cited_within("번호가 없다", cites) is False


# ── 백엔드 바꿔 끼우기 (labeling/llm.py 와 같은 모양) ────────────────


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    """설정을 시험이 정한 값으로만 만든다.

    **`os.environ` 만 지우면 안 된다** — `Settings` 가 `.env` 도 읽으므로 개발자가
    `.env` 에 키를 채우면 시험이 그 값을 보고 깨진다 (2026-09-16 실측). `_env_file=None`
    으로 `.env` 를 아예 읽지 않게 하고, 쓰는 자리의 `get_settings` 를 갈아 끼운다.
    """
    from family_fitness_ai.coach import llm
    from family_fitness_ai.common.settings import Settings

    # 프로세스 환경과 `ant` 프로필도 떼어낸다 — 그것이 있는 기계에서는
    # 「키가 없으면 드러난다」 시험이 조용히 통과해 버린다.
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm, "_has_ambient_credential", lambda: False)

    def apply(**values: str) -> None:
        settings = Settings(_env_file=None, **values)  # type: ignore[arg-type]
        monkeypatch.setattr(llm, "get_settings", lambda: settings)

    apply()
    return apply


def test_꺼져_있으면_발췌다(env) -> None:  # noqa: ANN001
    from family_fitness_ai.coach.llm import writer_from_settings

    env()
    assert writer_from_settings() is None


def test_백엔드마다_다른_쓰기를_준다(env) -> None:  # noqa: ANN001
    from family_fitness_ai.coach.llm import (
        DEFAULT_CLAUDE_MODEL,
        DEFAULT_GEMINI_MODEL,
        writer_from_settings,
    )

    env(coach_llm="true", anthropic_api_key="시험용")
    claude = writer_from_settings()
    assert claude is not None and claude.version() == f"claude:{DEFAULT_CLAUDE_MODEL}"

    env(coach_llm="true", coach_backend="gemini", gemini_api_key="시험용")
    gemini = writer_from_settings()
    assert gemini is not None and gemini.version() == f"gemini:{DEFAULT_GEMINI_MODEL}"


def test_모델을_바꿔_끼울_수_있다(env) -> None:  # noqa: ANN001
    from family_fitness_ai.coach.llm import writer_from_settings

    env(coach_llm="true", coach_backend="gemini", gemini_api_key="키", coach_model="딴모델")
    writer = writer_from_settings()
    assert writer is not None and writer.version() == "gemini:딴모델"


@pytest.mark.parametrize(
    ("backend", "needs"),
    [("claude", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")],
)
def test_키가_없으면_조용히_발췌로_가지_않는다(env, backend: str, needs: str) -> None:  # noqa: ANN001
    """켜 두고 발췌를 내보내면 운영자는 LLM 이 도는 줄 안다."""
    from family_fitness_ai.coach.llm import WriterUnavailable, writer_from_settings

    env(coach_llm="true", coach_backend=backend)
    with pytest.raises(WriterUnavailable, match=needs):
        writer_from_settings()


def test_두_백엔드가_같은_프로토콜을_만족한다() -> None:
    """호출부는 `write()` 하나만 본다 — 새 백엔드를 더해도 `answer` 는 바뀌지 않는다."""
    from family_fitness_ai.coach.llm import ClaudeWriter, GeminiWriter

    for writer in (ClaudeWriter(), GeminiWriter()):
        assert callable(writer.write)
        assert isinstance(writer.version(), str)
        assert writer.version()


def test_gemini_가_같은_프롬프트를_쓴다() -> None:
    """근거 번호가 인용 번호와 같아야 `[n]` 검사를 통과한다 — 두 백엔드가 같은 프롬프트다."""
    from family_fitness_ai.coach.llm import GeminiWriter, build_prompt

    hits = [_hit(1), _hit(2)]
    assert GeminiWriter().version().startswith("gemini:")
    prompt = build_prompt("물음", hits)
    assert "[1]" in prompt and "[2]" in prompt
