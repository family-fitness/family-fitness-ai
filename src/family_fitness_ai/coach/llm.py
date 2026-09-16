"""찾은 청크로 답 문구를 쓴다 (docs/01 §3.1 — LLM 호출은 여기 한 곳뿐이다).

**LLM 은 문구만 쓴다** (AGENTS.md §7). 무엇을 근거로 삼을지는 검색이 정하고, 이
모듈은 그 청크들을 사람이 읽을 한 단락으로 옮긴다. 근거를 고르거나 더하지 않는다.

**지어낼 자리를 주지 않는다** — 프롬프트에 청크 본문만 넣고, 그 밖의 지식을 쓰지
말라고 적고, 나온 답을 `docs/04` §4.2 검사로 다시 건다. 검사를 통과하지 못하면
발췌로 강등한다 (`docs/01` §3.1 — 「실패 시 규칙 편성으로 강등」).

백엔드가 `coach/messages` 에 주는 예산은 **10초 · 재시도 0회**다 (`AiGateway`).
그래서 마감을 두고 넘으면 강등한다 — 재시도하지 않는다 (중복 과금).

기본은 꺼져 있다 (`COACH_LLM`). 켜지 않으면 발췌만 나가고 외부 호출이 0이다.

**두 백엔드를 `COACH_BACKEND` 로 바꿔 끼운다** — `claude` · `gemini`.
`labeling/llm.py` 가 `LABELER_BACKEND` 로 라마·클로드를 바꾸는 것과 같은 모양이고,
호출부는 `AnswerWriter` 프로토콜 하나만 본다. 고른 쪽의 키가 없으면 기동이 아니라
첫 호출에서 503 으로 드러난다 — 조용히 발췌로 가지 않는다.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import anthropic
from anthropic.types.beta import (
    BetaMessageParam,
    BetaOutputConfigParam,
    BetaTextBlock,
)

from ..common.settings import get_settings

# `labeling/llm.py` 와 같은 값을 쓴다 — 모델이 두 벌이 되면 한쪽만 갱신된다.
from ..labeling.llm import CLAUDE_FALLBACK_BETA, DEFAULT_CLAUDE_MODEL
from ..rag.search import Hit

# 답은 두세 문장이다. 길게 쓸 이유가 없고, 길면 10초 예산을 넘긴다.
MAX_TOKENS = 2000

# 백엔드가 `coach/messages` 에 주는 예산은 10초다 (`AiGateway`). 그 안에서 끝내야
# 우리가 발췌로 강등할 틈이 남는다 — 3초를 검색·직렬화 몫으로 둔다.
DEADLINE_SEC = 7.0

# **Gemini 는 7초를 거부한다** — 「Manually set deadline 7s is too short. Minimum
# allowed deadline is 10s」 (2026-09-16 실측). 그 최소값이 백엔드 예산 전부와 같아서
# **Gemini 경로에는 강등할 틈이 없다**: 모델이 오래 끌면 우리가 발췌로 내려앉기 전에
# 백엔드가 먼저 끊는다. 예산을 넓히는 것은 계약 문제다 (docs/05).
GEMINI_DEADLINE_SEC = 10.0

# 짧은 인용 답변이라 깊이 생각할 일이 아니다. **생각을 끄지는 않는다** —
# Opus 5 에서 끄면 도구 호출을 본문에 쓰거나 내부 태그가 새는 일이 있다.
EFFORT: Literal["low", "medium", "high", "xhigh", "max"] = "low"

# Gemini 쪽 기본 모델 (2026-09-16 지정). 짧은 인용 답변이라 flash 급이다.
# 이 키로 무엇을 부를 수 있는지는 `client.models.list()` 로 확인한다 — 못 부르는
# 모델이면 `COACH_MODEL` 로 바꾼다.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# `EFFORT` 와 같은 뜻의 Gemini 값. 둘이 같은 낮은 단계를 가리킨다.
# `types.ThinkingLevel` 의 값이다 (SDK 2.23.0: MINIMAL·LOW·MEDIUM·HIGH).
GEMINI_THINKING: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"] = "LOW"

SYSTEM = """너는 국민체력100 자료만 근거로 답하는 체력 코치다.

**근거는 대개 빈도표다** — 「또래 몇 명에게 무엇이 몇 % 처방됐다」거나 영상의 구간
목록이다. 산문이 아니라고 답할 수 없는 것이 아니다. 「무엇을 하면 되나」 류의 물음에는
**그 표에서 많이 나온 것을 그대로 옮겨** 답한다.

지켜야 할 것
- **주어진 근거 안에서만 답한다.** 근거에 없는 내용을 덧붙이지 않는다
- 문장 끝마다 그 문장의 근거 번호를 `[n]` 으로 박는다. 번호는 주어진 것만 쓴다
- 두세 문장으로 짧게. 목록이나 제목을 쓰지 않는다
- **의료 표현을 쓰지 않는다** — 진단·치료·처방으로 읽히는 말을 하지 않는다.
  「처방」은 공단 자료의 운동 추천을 부르는 말일 때만 쓴다
- 「부족」·「미달」·「하위」 같은 말을 쓰지 않는다. 순위가 아니라 활동으로 말한다
- 근거가 물음과 **아예 다른 주제**일 때만 `근거없음` 한 낱말을 쓴다. 표라서 읽기
  어려운 것은 그 사유가 아니다 — 옮겨 적을 수 있으면 답한다"""

NO_BASIS = "근거없음"


class AnswerWriter(Protocol):
    def version(self) -> str: ...

    def write(self, question: str, hits: Sequence[Hit]) -> str: ...

    def write_copy(self, system: str, prompt: str) -> str:
        """문구 한 덩이. `write` 와 달리 근거를 넘기지 않는다 — 부르는 쪽이 프롬프트를
        다 만든다. 미션 카드 문구가 이것을 쓴다."""
        ...


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    """근거를 번호와 함께 싣는다. 번호는 인용 번호와 같다 (1부터)."""
    basis = "\n\n".join(
        f"[{number}] {hit.citation_label}\n{hit.text}" for number, hit in enumerate(hits, start=1)
    )
    return f"근거\n\n{basis}\n\n물음\n{question}"


@dataclass
class ClaudeWriter:
    """Anthropic API.

    **키를 명시로 넘긴다.** `.env` 는 `pydantic-settings` 가 읽지만 `os.environ` 으로
    내보내지 않으므로, 키를 넘기지 않으면 SDK 가 `.env` 의 값을 보지 못하고
    「Could not resolve authentication method」로 터진다 (2026-09-16 실측).
    `api_key` 가 `None` 이면 SDK 가 환경·`ant` 프로필에서 찾는다.
    """

    model: str = DEFAULT_CLAUDE_MODEL
    # **`repr` 에서 뺀다.** dataclass 의 기본 `repr` 은 값을 그대로 찍어서, 단정문
    # 실패 메시지·로그·예외에 키가 실린다 (2026-09-16 실측 — 시험 출력에 찍혔다).
    api_key: str | None = field(default=None, repr=False)
    # 비우면 부를 때 만든다. **시험에서 바꿔 끼운다** — 시험이 외부를 부르지 않는다.
    client: anthropic.Anthropic | None = None

    def version(self) -> str:
        return f"claude:{self.model}"

    def write(self, question: str, hits: Sequence[Hit]) -> str:
        return self.write_copy(SYSTEM, build_prompt(question, hits))

    def write_copy(self, system: str, prompt: str) -> str:
        # 재시도 0 — 백엔드가 10초 안에 한 번만 부른다 (중복 과금을 만들지 않는다)
        client = self.client or anthropic.Anthropic(
            api_key=self.api_key, timeout=DEADLINE_SEC, max_retries=0
        )
        messages: list[BetaMessageParam] = [{"role": "user", "content": prompt}]
        output_config: BetaOutputConfigParam = {"effort": EFFORT}
        response = client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            betas=[CLAUDE_FALLBACK_BETA],
            fallbacks="default",
            system=system,
            output_config=output_config,
            messages=messages,
        )
        # 거절은 내용보다 먼저 본다 — 대체 모델까지 거절했다는 뜻이다
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise WriterFailed(f"거절 · {category}")
        if response.stop_reason == "max_tokens":
            raise WriterFailed("응답이 잘렸다")
        text = "".join(
            block.text for block in response.content if isinstance(block, BetaTextBlock)
        ).strip()
        if not text:
            raise WriterFailed("빈 응답")
        return text


@dataclass
class GeminiWriter:
    """Google Gemini. `ClaudeWriter` 와 **같은 자리에 끼워진다** (`AnswerWriter`).

    `labeling/llm.py` 가 `LABELER_BACKEND` 로 라마·클로드를 바꿔 끼우는 것과 같은
    모양이다 — 호출부는 `write()` 하나만 본다.

    SDK 는 `google-genai` 다. 아래 값은 **설치된 2.23.0 과 실호출에서 확인한 것**이다 —
    `HttpOptions.timeout` 은 밀리초(초가 아니다), 그 **최소값이 10초**이고,
    `ThinkingLevel` 은 MINIMAL·LOW·MEDIUM·HIGH, 잘림·차단은
    `candidates[0].finish_reason` 으로 본다 (`STOP` 이 아니면 온전한 답이 아니다).
    """

    model: str = DEFAULT_GEMINI_MODEL
    # `ClaudeWriter` 와 같은 이유로 `repr` 에서 뺀다 — 키가 로그에 실리지 않게.
    api_key: str | None = field(default=None, repr=False)
    # 비우면 부를 때 만든다. **시험에서 바꿔 끼운다.**
    client: object | None = None

    def version(self) -> str:
        return f"gemini:{self.model}"

    def write(self, question: str, hits: Sequence[Hit]) -> str:
        return self.write_copy(SYSTEM, build_prompt(question, hits))

    def write_copy(self, system: str, prompt: str) -> str:
        from google.genai import Client, types

        client = self.client or Client(
            api_key=self.api_key,
            # 재시도 0 — 백엔드가 10초 안에 한 번만 부른다 (중복 과금을 만들지 않는다).
            # `attempts=1` 이 「한 번만 보낸다」다 (0 이 아니다).
            http_options=types.HttpOptions(
                # 밀리초다 (SDK 2.23.0 확인). 10초는 Gemini 가 받는 **최소값**이다
                timeout=int(GEMINI_DEADLINE_SEC * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        response = client.models.generate_content(  # type: ignore[attr-defined]
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=MAX_TOKENS,
                # 짧은 인용 답변이라 깊이 생각할 일이 아니다. 끄지 않고 낮춘다 —
                # 끄면 근거를 대충 읽고 번호를 틀리게 박는다
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel(GEMINI_THINKING)
                ),
                # 도구를 주지 않으므로 끈다. 켜 두면 SDK 가 요청마다 경고를 찍는다
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        candidate = (response.candidates or [None])[0]
        reason = getattr(candidate, "finish_reason", None)
        # 차단·잘림은 내용보다 먼저 본다. `STOP` 이 아니면 온전한 답이 아니다
        if reason is not None and str(getattr(reason, "name", reason)) not in ("STOP",):
            raise WriterFailed(f"끝난 이유가 STOP 이 아니다 · {reason}")
        text = (response.text or "").strip()
        if not text:
            raise WriterFailed("빈 응답")
        return text


class WriterFailed(RuntimeError):
    """문구를 쓰지 못했다. **거부가 아니라 강등이다** — 발췌로 답한다."""


class WriterUnavailable(RuntimeError):
    """설정이 켜져 있는데 쓸 수 없다. **강등이 아니라 설정 오류다.**

    켜 두고 조용히 발췌를 내보내면 운영자는 LLM 이 도는 줄 안다 — 「데이터가 없을 때
    더미로 채워 돌아가게 만들지 않는다」 (AGENTS.md §4). 기동·첫 호출에서 드러낸다.
    """


def writer_from_settings() -> AnswerWriter | None:
    """`COACH_BACKEND` 하나로 바꿔 끼운다. 호출부는 같다 (`labeling/llm.py` 와 같은 모양).

    `COACH_LLM` 이 꺼져 있으면 `None` 이다 — 호출부가 발췌로 간다.
    켜져 있는데 자격증명이 없으면 `WriterUnavailable` 이다. **조용히 발췌로 가지 않는다.**
    """
    settings = get_settings()
    if not settings.coach_llm:
        return None

    if settings.coach_backend == "gemini":
        key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not key and not os.environ.get("GOOGLE_API_KEY"):
            raise WriterUnavailable(
                "COACH_BACKEND=gemini 인데 자격증명이 없다 — "
                "GEMINI_API_KEY 를 .env 나 환경에 넣는다"
            )
        return GeminiWriter(model=settings.coach_model or DEFAULT_GEMINI_MODEL, api_key=key)

    key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key and not _has_ambient_credential():
        raise WriterUnavailable(
            "COACH_BACKEND=claude 인데 자격증명이 없다 — "
            "ANTHROPIC_API_KEY 를 .env 나 환경에 넣거나 `ant auth login` 을 한다"
        )
    return ClaudeWriter(model=settings.coach_model or DEFAULT_CLAUDE_MODEL, api_key=key)


def _has_ambient_credential() -> bool:
    """키 말고도 SDK 가 찾을 자격증명이 있나 — `ANTHROPIC_AUTH_TOKEN` 또는 `ant` 프로필.

    프로필 디렉터리가 있는지만 본다. 값을 읽지 않는다.
    """
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


# ── 미션 카드 문구 ──────────────────────────────────────────────────

MISSION_SYSTEM = """너는 가족 운동 미션 카드의 문구를 쓴다.

**주어진 운동과 근거만 쓴다.** 운동을 바꾸거나 더하거나 빼지 않는다 — 무엇을 할지는
이미 정해졌고 너는 그것을 사람이 읽을 말로 옮긴다.

- `child` 는 아이에게 말한다. 한 문장. 다정하고 짧게. 명령하지 않는다
- `parent` 는 보호자에게 말한다. 한두 문장. **근거를 말한다** — 또래 처방에 얼마나
  나온 운동인지. 시간(분)과 주 몇 회인지를 함께
- **주어지지 않은 수를 지어내지 않는다.** 주당 횟수는 위에 적힌 것만 쓴다.
  하루짜리 카드에는 **주당 횟수를 아예 쓰지 않는다** — 오늘 할 것만 말한다
- **숫자를 바꾸거나 섞지 않는다.** 「순위」와 「처방된 횟수」는 다른 값이다 —
  순위를 횟수처럼 쓰거나 횟수를 운동 횟수로 쓰지 않는다. 헷갈리면 **숫자를 빼고**
  「또래 처방에 많이 나온 운동」이라고만 쓴다
- **그 운동이 무엇을 기른다고 말하지 않는다.** 재지 않았다. 말할 수 있는 것은
  「또래 처방에 이만큼 나왔다」뿐이다
- **의료 표현을 쓰지 않는다** — 진단·치료로 읽히는 말을 하지 않는다
- 「부족」·「미달」·「하위」 같은 말을 쓰지 않는다. 순위가 아니라 활동으로 말한다
- `[n]` 같은 인용 번호를 문구에 넣지 않는다 — 카드에는 근거가 따로 붙는다

두 줄로만 답한다. 다른 말을 덧붙이지 않는다.

child: <아이 문구>
parent: <보호자 문구>"""


def mission_prompt(
    who: str,
    exercises: Sequence[tuple[str, str, int, int]],
    minutes: int,
    daily: bool,
    days_per_week: int,
) -> str:
    """`exercises` 는 `(운동 이름, 단계, 처방 순위, 처방 횟수)` 다.

    **주당 횟수를 반드시 넘긴다.** 일일 미션에 그것을 주지 않으면 모델이 지어낸다 —
    주 3회인데 「주 7회 함께 해 주세요」가 나왔다 (2026-09-16 실측).
    """
    # **두 숫자를 헷갈리지 않게 이름을 붙인다.** 「1위 · 4,202회」처럼 나란히 두면
    # 모델이 순위를 횟수로 읽는다 (2026-09-16 실측 — 「28회 등장」·「534회를 진행」).
    lines = [
        f"- {name}\n"
        f"    단계: {phase}\n"
        f"    또래 처방에서 몇 번째로 많이 나왔나: {rank}번째\n"
        f"    또래 처방에 나온 횟수: {count:,}번"
        for name, phase, rank, count in exercises
    ]
    kind = "오늘 하루" if daily else f"이번 주 {len(exercises)}회"
    return f"대상: {who}\n기간: {kind}\n회당: {minutes}분\n운동\n" + "\n".join(lines)


# 「주 3회」류. 문구가 주당 횟수를 말했다면 그 수가 맞아야 한다.
WEEKLY_CLAIM = re.compile(r"주\s*(\d+)\s*회")


def weekly_claim_ok(text: str, *, daily: bool, days_per_week: int) -> bool:
    """문구가 말한 주당 횟수가 맞나. **말하지 않았으면 통과다.**

    프롬프트로 막아도 새는 값이라 검사로 잡는다 — 일일 카드에 「주 1회」, 주 3회인데
    「주 7회」가 실렸다 (2026-09-16 실측). 틀리면 규칙 문구로 강등한다.
    """
    claims = {int(n) for n in WEEKLY_CLAIM.findall(text)}
    if not claims:
        return True
    # 일일 카드는 주당 횟수를 말할 자리가 아니다 — 이 카드는 하루치다
    return not daily and claims == {days_per_week}


def parse_copy(text: str) -> tuple[str, str]:
    """`child:` · `parent:` 두 줄을 읽는다. 하나라도 없으면 `WriterFailed` 다."""
    child = parent = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("child:"):
            child = stripped[len("child:") :].strip()
        elif stripped.startswith("parent:"):
            parent = stripped[len("parent:") :].strip()
    if not child or not parent:
        raise WriterFailed(f"child·parent 두 줄을 찾지 못했다: {text[:80]!r}")
    return child, parent
