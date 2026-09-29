"""환경 설정. 값은 .env 에서 오고, 이름만 .env.example 에 둔다."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    #: 임베딩을 어디서 돌리나. local 은 이 프로세스 안(llama-cpp-python), http 는
    #: 밖에 띄운 서버다. 비우면 local. 어느 쪽이든 인덱스를 만든 모델과 같아야 한다.
    embedding_backend: str = "local"
    #: local 이 읽을 모델 파일. 비우면 허깅페이스 캐시의 gpustack/bge-m3-GGUF 다.
    embedding_gguf: str = ""
    #: http 일 때만 쓴다. llama.cpp 서버 (bge-m3).
    embedding_url: str = "http://127.0.0.1:8082"
    #: 검색이 이 아래면 근거로 치지 않는다.
    sim_threshold: float = 0.57

    index_dir: Path = ROOT / "data" / "index"
    release_dir: Path = ROOT / "data" / "release"
    raw_dir: Path = ROOT / "data" / "raw"

    #: 편성과 문구를 LLM 이 맡나. 기본은 켜짐이고, 키가 없거나 호출이 실패하면
    #: 규칙 편성으로 내려간다 — 서비스가 멈추지는 않는다.
    coach_llm: bool = True
    #: claude | gemini. 아래 모델 이름이 백엔드를 말해 주면 그쪽을 따른다.
    coach_backend: str = "claude"
    #: 비우면 백엔드의 기본 모델. 이름만 바꿔도 백엔드가 따라간다 —
    #: `claude-haiku-4-5` · `claude-sonnet-5` · `gemini-flash-latest` 다 된다.
    coach_model: str = ""

    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    @field_validator("coach_llm", mode="before")
    @classmethod
    def _empty_is_default(cls, value: object) -> object:
        """.env 는 이름만 적고 값을 비워 두는 일이 잦다. 빈 값은 기본값이다."""
        return True if value == "" else value


@lru_cache
def settings() -> Settings:
    return Settings()
