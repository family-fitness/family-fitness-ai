"""data/release 표. 서비스가 읽는 여섯 장이다 — stats.build · video.clips · video.labels ·
video.kspo 가 만들고, 저장소에 들어 있다.

인덱스(rag.index.FILES)와 같이 띄울 때 있는지 본다. 한 장이라도 없으면 서버는 떠도
평가가 까닭 없는 503 을 내거나(value_quantiles · grade_*), 영상 후보가 소리 없이
줄어든다(kspo_videos 가 없으면 공단 영상이 모두 빠진다).
"""

from __future__ import annotations

from pathlib import Path

from family_fitness_ai.common.settings import settings

#: 평가 · 궤적이 읽는 셋(stats.tables)과 편성 · 영상 찾기가 읽는 셋(video.catalog).
FILES = (
    "value_quantiles.csv",
    "grade_thresholds.csv",
    "grade_distribution.csv",
    "video_clips.csv",
    "clip_labels.csv",
    "kspo_videos.csv",
)


def missing_files(directory: Path | None = None) -> list[str]:
    """없는 release 표 이름. 비어 있으면 갖춰진 것이다."""
    directory = directory or settings().release_dir
    return [name for name in FILES if not (directory / name).exists()]
