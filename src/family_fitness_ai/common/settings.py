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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

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
    # docs/04 §3 — 근거 없는 임계값은 조용히 관련 자료를 버린다. 기본값을 두지 않는다.
    sim_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    def configured(self) -> dict[str, bool]:
        """이름과 설정 여부만 돌려준다. **값을 로그에 찍지 않는다** (docs/01 §5)."""
        return {
            name: getattr(self, name) is not None
            for name in self.__class__.model_fields
            if name != "vector_backend"
        }


# 산출물과 인덱스의 자리. 환경변수로 가르지 않는다 — 저장소 안의 경로다.
RELEASE_DIR = Path("data/release")
INDEX_DIR = Path("data/index")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """기동 시 한 번. 캐시가 '한 번 읽는다'를 강제한다."""
    return Settings()
