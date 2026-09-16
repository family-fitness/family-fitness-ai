"""환경변수를 타입 있는 객체 하나로 읽는다 (docs/01 §3.2).

기동 시 한 번 읽고 검증한다. 요청마다 `os.environ` 을 보지 않는다.
**값은 이미지에도 코드에도 없다** — 배포에서는 Secrets Manager 와 SSM 이 주입한다.

없어도 되는 값과 없으면 안 되는 값을 나눈다. `SIM_THRESHOLD` 는 후자지만 여기서
막지 않는다 — 비어 있으면 `/readyz` 가 준비되지 않았다고 답한다 (docs/04 §6 ②).
기동 자체를 막으면 무엇이 빠졌는지 로그를 뒤져야 알 수 있다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

VectorBackend = Literal["pgvector", "faiss"]
LabelerBackend = Literal["llama", "claude"]

# 기본값이 있어 "설정했는가"가 뜻이 없는 이름. `configured()` 에 싣지 않는다.
_NOT_REPORTED = frozenset(
    {"vector_backend", "labeler_backend", "llama_server_url", "embedding_url"}
)


class Settings(BaseSettings):
    # 빈 값은 설정하지 않은 것으로 본다. `.env.example` 을 복사하면 `SIM_THRESHOLD=` 처럼
    # 빈 줄이 남는데, 그것을 숫자로 읽으려다 기동이 막히면 안 된다.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False, env_ignore_empty=True
    )

    # 비밀값 — 없어도 기동한다. 쓰는 시점에 없으면 그 기능이 실패한다.
    database_url: str | None = None
    anthropic_api_key: str | None = None
    youtube_api_key: str | None = None
    kspo_service_key: str | None = None
    kspo_endpoint: str | None = None

    # 설정
    mlflow_tracking_uri: str | None = None
    # docs/01 §1 — DB가 서기 전에도 개발이 멈추지 않게 한다. 배포 기본값은 pgvector 다.
    vector_backend: VectorBackend = "faiss"
    # docs/04 §3 — 잰 값은 0.50 (docs/dev/AI-8 §5). 여기에 기본값을 두지 않는 이유는,
    # 임베딩 모델을 바꾸고도 다시 재지 않은 채 서비스가 뜨는 길을 막기 위해서다.
    sim_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    # docs/dev/AI-7 §3.8 — 영상 라벨 LLM. 기본은 외부 호출이 없는 로컬이다 (AGENTS.md §3).
    labeler_backend: LabelerBackend = "llama"
    llama_server_url: str = "http://127.0.0.1:8081"
    # docs/dev/AI-8 §1.3 — 청크·질의 임베딩 (`bge-m3` · 1024). 색인과 검색이 같은 것을 쓴다.
    embedding_url: str = "http://127.0.0.1:8082"
    # 비우면 백엔드의 기본 모델이다.
    labeler_model: str | None = None

    def configured(self) -> dict[str, bool]:
        """이름과 설정 여부만 돌려준다. **값을 로그에 찍지 않는다** (docs/01 §5)."""
        return {
            name: getattr(self, name) is not None
            for name in self.__class__.model_fields
            if name not in _NOT_REPORTED
        }


# 산출물과 인덱스의 자리. 환경변수로 가르지 않는다 — 저장소 안의 경로다.
RELEASE_DIR = Path("data/release")
INDEX_DIR = Path("data/index")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """기동 시 한 번. 캐시가 '한 번 읽는다'를 강제한다."""
    return Settings()
