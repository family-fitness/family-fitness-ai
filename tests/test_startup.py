"""띄울 때 보는 것. 갖춰지지 않았으면 뜨지 않아야 배포가 실패한 줄 바로 안다."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from family_fitness_ai.api import app as api
from family_fitness_ai.common import release
from family_fitness_ai.common.settings import settings


def _start() -> None:
    """lifespan 에 들어갔다 나온다. 뜨지 못하면 그 예외가 그대로 올라온다."""

    async def run() -> None:
        async with api.lifespan(api.app):
            pass

    asyncio.run(run())


@pytest.fixture
def ready(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """인덱스와 임베딩은 갖춰졌다고 친다. 여기서는 release 표만 본다."""
    monkeypatch.setattr(api, "missing_files", lambda: [])
    monkeypatch.setattr(api.embed, "missing", lambda: "")
    monkeypatch.setattr(settings(), "embedding_warmup", False)
    return monkeypatch


def _release_without(tmp_path: Path, *absent: str) -> Path:
    for name in release.FILES:
        if name not in absent:
            (tmp_path / name).write_text("", encoding="utf-8")
    return tmp_path


def test_it_starts_when_every_release_table_is_there(ready, tmp_path):
    ready.setattr(settings(), "release_dir", _release_without(tmp_path))
    _start()


def test_it_refuses_to_start_without_the_release_tables(ready, tmp_path):
    ready.setattr(settings(), "release_dir", tmp_path)
    with pytest.raises(RuntimeError, match="value_quantiles.csv"):
        _start()


def test_a_missing_kspo_table_alone_stops_it(ready, tmp_path):
    """공단 영상 표 하나만 없어도 편성 후보에서 452편이 소리 없이 빠진다."""
    ready.setattr(settings(), "release_dir", _release_without(tmp_path, "kspo_videos.csv"))
    with pytest.raises(RuntimeError, match="kspo_videos.csv"):
        _start()


def test_the_repository_carries_every_release_table():
    assert release.missing_files() == []


def test_it_loads_the_embedding_model_before_taking_requests(ready, monkeypatch):
    warmed: list[bool] = []
    monkeypatch.setattr(settings(), "embedding_warmup", True)
    monkeypatch.setattr(api.embed, "warm_up", lambda: warmed.append(True))
    _start()
    assert warmed == [True]


def test_warming_up_can_be_turned_off(ready, monkeypatch):
    """--reload 로 코드를 고칠 때마다 모델을 기다리지 않게 끌 수 있다."""
    warmed: list[bool] = []
    monkeypatch.setattr(settings(), "embedding_warmup", False)
    monkeypatch.setattr(api.embed, "warm_up", lambda: warmed.append(True))
    _start()
    assert warmed == []
