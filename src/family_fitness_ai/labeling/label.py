"""영상 라벨 — 적혀 있는 것만 규칙으로 옮긴다 (docs/dev/AI-7 §3).

**LLM 보다 규칙이 먼저다** (AGENTS.md §7). 제목·재생목록·화면 이름표·설명문·자막에 적힌
낱말을 라벨로 옮기고, 적혀 있지 않으면 비운다. 추정하지 않는다.

실행 (collect·frames 뒤, 저장소 루트에서):
    python -m family_fitness_ai.labeling.label
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from ..common.types import AGE_GROUPS, FITNESS_FACTORS, AgeGroup, FitnessFactor
from ..rag.prescription import VOCABULARY_FILE, identity
from . import collect as C

# docs/01 §3.3. v2 는 화면 이름표를, v3 는 아래 이름 막대를 읽는다.
LABELER_VERSION = "labeler:rules/v3"

LABELS_FILE = "video_labels.csv"
EXERCISES_FILE = "video_exercises.csv"

# 서비스 대상이 아니다. 라벨하지 않는다 (docs/dev/AI-7 §3.5).
EXCLUDED_PLAYLISTS = frozenset({"PLBdpvOnWjVZlsOa_LGjvlta7sg3NBFbqs"})  # 연령별 맞춤 운동 : 중장년

# 연령대 이름과, 재생목록이 연령대 대신 쓰는 낱말 둘(`영유아`·`청년`)만 옮긴다.
# `유아`·`유치원생`·`초등학생` 은 옮기지 않는다 — 옮기면 추정이 된다 (docs/dev/AI-7 §3.1).
AGE_WORDS: dict[str, AgeGroup] = {
    "영유아": "유아기",
    "유아기": "유아기",
    "유소년": "유소년",
    "청소년": "청소년",
    "청년": "성인",
    "성인": "성인",
    "어르신": "어르신",
}

# 합치는 것은 표기 차이 둘뿐이다. `균형` 은 요인으로 읽지 않는다 — 제목에서
# 「균형있는 근육 발달」처럼 쓰인다.
FACTOR_WORDS: dict[str, FitnessFactor] = {
    **{f: f for f in FITNESS_FACTORS},
    "심폐체력": "심폐지구력",
    "협응성": "협응력",
}

# 공단 설명문의 해시태그는 영상마다 같은 상용구다. 읽지 않는다 (docs/dev/AI-7 §3.2).
HASHTAG = re.compile(r"#\S+")

# 믿는 순서. 같은 이름이 여러 곳에서 잡히면 앞의 것을 남긴다 (docs/dev/AI-7 §3.4).
SOURCES = ("title", "screen", "screen_bar", "description", "captions")

# 화면 이름표 (docs/dev/AI-7 §3.7). 윗줄에는 `편` 이 들어간다 — `(응용편) 민첩성·순발력·협응성`.
HEADER_MARK = "편"
# 한 프레임에만 읽힌 이름은 글자 인식 잡음이다.
MIN_FRAMES = 2
# 윗줄 요인은 구간 윗줄의 이 비율 이상에서 읽혀야 붙인다.
# `협응성` 을 `협성` 으로 읽는 프레임이 있다.
FACTOR_SHARE = 0.3

# 아래 이름 막대의 번호·느낌표와 괄호 속 다른 이름 (docs/dev/AI-7 §3.9).
_BAR_NUMBER = re.compile(r"^\s*\d+\s*[.)]\s*")
_BAR_ALIAS = re.compile(r"(.+?)\s*\((.+)\)")

# 같은 재생목록 이만큼의 영상에 똑같이 나오는 이름은 공통 준비·마무리다 (docs/dev/AI-7 §3.10).
# 실측: 영유아 4개 이름이 11편, 청소년 7개 이름이 5~6편, 그다음이 2편이라 3~5 어디든 같다.
COMMON_MIN_VIDEOS = 3

LABEL_COLUMNS = [
    "video_id",
    "title",
    "playlist_titles",
    "duration_sec",
    "age_group",
    "age_evidence",
    "fitness_factors",
    "factor_evidence",
    "exercise_names",
    "common_exercise_names",
    "start_sec",
    "space",
    "noise",
    "equipment",
    "intensity",
    "caption",
    "screen",
    "labeler_version",
]
EXERCISE_COLUMNS = [
    "exercise_name",
    "video_id",
    "source",
    "common",
    "evidence_text",
    "fitness_factors",
    "age_group",
    "start_sec",
    "url",
]

T = TypeVar("T")


def find_words(text: str, table: Mapping[str, T]) -> list[T]:
    """긴 낱말부터 찾고 찾은 자리는 지운다 — 짧은 낱말이 긴 낱말 안에서 또 잡히지 않게."""
    found: list[T] = []
    for word in sorted(table, key=len, reverse=True):
        if word in text:
            text = text.replace(word, " ")
            if table[word] not in found:
                found.append(table[word])
    return found


def age_of(title: str, playlist_titles: Iterable[str]) -> tuple[AgeGroup | None, str]:
    """제목·재생목록에 **적힌** 연령만 (docs/dev/AI-7 §3.1). 태그·설명문·자막은 보지 않는다.

    **제목에 적힌 연령이 재생목록보다 앞선다** — 「연령별 맞춤 운동 : 청소년」 재생목록에
    `[유소년]` 영상이 섞여 있다. 제목에 없을 때만 재생목록을 본다.

    같은 곳에 서로 다른 연령이 적혀 있으면 비운다 — 어느 쪽이 맞는지 규칙이 정할 수 없다.
    비워도 근거는 남긴다. 검수하는 사람이 왜 비었는지 봐야 한다.
    """
    for where, texts in (("제목", [title]), ("재생목록", list(playlist_titles))):
        found: dict[AgeGroup, None] = {}
        evidence: list[str] = []
        for text in texts:
            groups = find_words(text, AGE_WORDS)
            if groups:
                found.update(dict.fromkeys(groups))
                evidence.append(f"{where}: {text}")
        if found:
            group = next(iter(found)) if len(found) == 1 else None
            return group, " | ".join(evidence)
    return None, ""


@dataclass(frozen=True)
class Segment:
    """화면 이름표 하나가 떠 있던 구간."""

    name: str
    start_sec: int
    end_sec: int
    factors: tuple[FitnessFactor, ...]


def _screen_key(text: str) -> str:
    """이름표 비교용. 괄호·`#` 같은 글자 인식 부스러기를 뗀다."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", identity(text))


def screen_segments(frames: Iterable[Mapping[str, Any]], field: str = "texts") -> list[Segment]:
    """프레임마다 읽은 글자 → 구간 (docs/dev/AI-7 §3.7). `field` 는 읽은 자리다.

    윗줄이 아닌 마지막 줄이 이름이다. 같은 이름이 이어지는 동안이 한 구간이고, 이름표가
    잠깐 사라졌다 다시 떠도 사이에 다른 이름이 없으면 잇는다. 한 프레임에만 읽힌 이름은
    잡음으로 버린 뒤, 앞뒤의 같은 이름을 잇는다.
    """
    runs: list[dict[str, Any]] = []
    for frame in sorted(frames, key=lambda f: int(f["t"])):
        texts = [str(t) for t in frame.get(field, [])]
        names = [
            t
            for t in texts
            if HEADER_MARK not in t and len(_screen_key(t)) >= 3 and not t.isascii()
        ]
        if not names:
            continue
        name, key, t = names[-1], _screen_key(names[-1]), int(frame["t"])
        if not runs or runs[-1]["key"] != key:
            runs.append({"key": key, "start": t, "frames": 0, "names": Counter(), "headers": []})
        run = runs[-1]
        run["end"] = t
        run["frames"] += 1
        run["names"][name] += 1
        run["headers"] += [h for h in texts if HEADER_MARK in h]

    merged: list[dict[str, Any]] = []
    for run in (r for r in runs if r["frames"] >= MIN_FRAMES):
        if merged and merged[-1]["key"] == run["key"]:
            last = merged[-1]
            last["end"] = run["end"]
            last["frames"] += run["frames"]
            last["names"] += run["names"]
            last["headers"] += run["headers"]
        else:
            merged.append(run)
    return [
        Segment(
            name=run["names"].most_common(1)[0][0].lstrip("#").strip(),
            start_sec=run["start"],
            end_sec=run["end"],
            factors=_vote(run["headers"]),
        )
        for run in merged
    ]


def _vote(headers: list[str]) -> tuple[FitnessFactor, ...]:
    """윗줄 요인의 다수결. `기본운동기술 안정성·이동성·조작성` 은 요인이 아니라 비어 나온다."""
    counts = Counter(f for h in headers for f in find_words(h, FACTOR_WORDS))
    need = max(MIN_FRAMES, FACTOR_SHARE * len(headers))
    return tuple(f for f in FITNESS_FACTORS if counts[f] >= need)


def _mmss(sec: int) -> str:
    return f"{sec // 60:02d}:{sec % 60:02d}"


def factors_of(
    title: str, segments: Iterable[Segment] = ()
) -> tuple[tuple[FitnessFactor, ...], str]:
    """제목과 화면 이름표 윗줄에 적힌 요인. 설명문은 보지 않는다 (docs/dev/AI-7 §3.2)."""
    found = set(find_words(title, FACTOR_WORDS))
    evidence = [f"제목: {title}"] if found else []
    for s in segments:
        if s.factors:
            found.update(s.factors)
            evidence.append(f"화면 {_mmss(s.start_sec)} {s.name}: {'·'.join(s.factors)}")
    return tuple(f for f in FITNESS_FACTORS if f in found), " | ".join(evidence)


class Vocabulary:
    """[AI-6] 처방 어휘. **이 안에서만 고른다** (docs/dev/AI-7 §3.3)."""

    def __init__(self, names: Iterable[str]) -> None:
        self._exact = {identity(n): n for n in names if identity(n)}
        # 긴 이름부터 — `줄넘기 운동` 이 `줄넘기` 로 한 번 더 잡히지 않게
        self._keys = sorted(self._exact.items(), key=lambda kv: (-len(kv[0]), kv[0]))

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        return cls(pd.read_csv(path, encoding="utf-8-sig", dtype=str)["exercise_name"])

    def __len__(self) -> int:
        return len(self._keys)

    def exact(self, text: str) -> str | None:
        """표기 차이까지만 같은 이름. 부분 일치는 보지 않는다."""
        return self._exact.get(identity(text))

    def find(self, normalized: str) -> list[tuple[str, int, int]]:
        """정규화한 본문에서 겹치지 않게 찾는다. `(이름, 시작, 끝)` — 정규화 좌표다."""
        taken = [False] * len(normalized)
        found: list[tuple[str, int, int]] = []
        for key, name in self._keys:
            start = normalized.find(key)
            while start != -1:
                end = start + len(key)
                if not any(taken[start:end]):
                    taken[start:end] = [True] * len(key)
                    found.append((name, start, end))
                    break
                start = normalized.find(key, start + 1)
        return sorted(found, key=lambda f: f[1])


@dataclass(frozen=True)
class Match:
    name: str
    source: str  # SOURCES 중 하나
    evidence: str  # 잡힌 줄 원문
    start_sec: int | None = None  # 화면에서만
    factors: tuple[FitnessFactor, ...] = ()  # 화면 이름표 윗줄에서만
    common: bool = False  # 여러 영상에 똑같이 나오는 준비·마무리 (§3.10)


def match_lines(vocab: Vocabulary, text: str, source: str) -> list[Match]:
    """줄마다 찾는다. 설명문은 한 줄이 한 동작이라 그 줄이 곧 근거다."""
    return [
        Match(name, source, line.strip())
        for line in text.splitlines()
        for name, _, _ in vocab.find(identity(line))
    ]


def match_screen(vocab: Vocabulary, segments: Iterable[Segment]) -> list[Match]:
    """이름표 이름이 어휘와 같을 때. 근거는 `시각 이름표` 이고 시작 시각과 윗줄 요인을 싣는다."""
    return [
        Match(name, "screen", f"{_mmss(s.start_sec)} {s.name}", s.start_sec, s.factors)
        for s in segments
        for name, _, _ in vocab.find(identity(s.name))
    ]


def bar_names(text: str) -> list[str]:
    """이름 막대 한 줄에서 어휘와 견줄 이름. 번호·느낌표를 떼고 괄호 속 이름도 따로 낸다.

    `3. 넙다리 안쪽 늘리기 (나비자세)!` → 통째, `넙다리 안쪽 늘리기`, `나비자세`
    """
    name = _BAR_NUMBER.sub("", text).strip().rstrip("!?.~ ")
    alias = _BAR_ALIAS.fullmatch(name)
    return [name, alias.group(1), alias.group(2)] if alias else [name]


def match_bar(vocab: Vocabulary, segments: Iterable[Segment]) -> list[Match]:
    """아래 이름 막대 (docs/dev/AI-7 §3.9). **이름이 어휘와 통째로 같을 때만** 붙인다.

    같은 자리에 설명·안전 문장도 뜬다 (`사이드 스텝으로 활동한다`). 부분 일치를 허용하면
    문장 속 낱말이 운동 이름이 된다. 통째가 어휘에 없을 때만 괄호 속 이름을 따로 본다.
    """
    out: list[Match] = []
    for s in segments:
        whole, *parts = bar_names(s.name)
        names = [vocab.exact(whole)] if vocab.exact(whole) else [vocab.exact(p) for p in parts]
        out += [
            Match(name, "screen_bar", f"{_mmss(s.start_sec)} {s.name}", s.start_sec)
            for name in names
            if name
        ]
    return out


def match_captions(vocab: Vocabulary, snippets: list[dict[str, Any]]) -> list[Match]:
    """자막은 이름이 조각 경계에서 끊긴다 (`거북이` | `스트레칭`). 이어 붙여 찾고,
    근거는 걸친 조각들의 원문이다."""
    texts = [str(s["text"]) for s in snippets]
    parts = [identity(t) for t in texts]
    bounds: list[tuple[int, int]] = []
    pos = 0
    for part in parts:
        bounds.append((pos, pos + len(part)))
        pos += len(part)
    out: list[Match] = []
    for name, start, end in vocab.find("".join(parts)):
        covered = [t for t, (a, b) in zip(texts, bounds, strict=True) if a < end and start < b]
        out.append(Match(name, "captions", " ".join(covered)))
    return out


def exercise_matches(
    vocab: Vocabulary,
    title: str,
    description: str,
    caption: Mapping[str, Any] | None,
    segments: Iterable[Segment] = (),
    bar: Iterable[Segment] = (),
) -> list[Match]:
    """이름마다 가장 믿을 만한 곳 하나 (docs/dev/AI-7 §3.4).

    제목에서 잡힌 이름도 시각과 요인은 화면에서 가져온다 — 그 둘은 화면에만 있다.
    """
    on_screen = [*match_screen(vocab, segments), *match_bar(vocab, bar)]
    found = [
        *match_lines(vocab, title, "title"),
        *on_screen,
        *match_lines(vocab, HASHTAG.sub(" ", description), "description"),
        *(match_captions(vocab, caption["snippets"]) if caption and "snippets" in caption else []),
    ]
    best: dict[str, Match] = {}
    for m in found:  # SOURCES 순서로 들어온다 — 먼저 온 것이 남는다
        best.setdefault(m.name, m)
    timed: dict[str, Match] = {}
    for m in on_screen:  # 이름표 먼저, 같은 이름이 여러 구간에 뜨면 처음 구간
        timed.setdefault(m.name, m)
    return [
        replace(m, start_sec=timed[m.name].start_sec, factors=timed[m.name].factors)
        if m.name in timed
        else m
        for m in best.values()
    ]


@dataclass(frozen=True)
class VideoLabel:
    video_id: str
    title: str
    playlist_titles: str
    duration_sec: str
    caption: str
    screen: str
    age_group: AgeGroup | None
    age_evidence: str
    fitness_factors: tuple[FitnessFactor, ...]
    factor_evidence: str
    exercises: tuple[Match, ...]


def screen_status(screen: Mapping[str, Any] | None, segments: list[Segment]) -> str:
    """빈칸은 안 읽음, 사유는 받을 수 없었음, 숫자는 읽은 이름표 구간 수."""
    if screen is None:
        return ""
    if "missing" in screen:
        return str(screen["missing"])
    return str(len(segments))


def mark_common(labels: list[VideoLabel]) -> list[VideoLabel]:
    """같은 재생목록의 여러 영상에 똑같이 나오는 이름을 공통 준비·마무리로 표시한다 (§3.10).

    **제목에서 잡힌 이름은 표시하지 않는다** — `EP03.거북이 스트레칭` 에서는 그것이 본 내용이다.
    """
    videos: dict[tuple[str, str], set[str]] = defaultdict(set)
    for v in labels:
        for m in v.exercises:
            videos[(v.playlist_titles, m.name)].add(v.video_id)
    return [
        replace(
            v,
            exercises=tuple(
                replace(
                    m,
                    common=m.source != "title"
                    and len(videos[(v.playlist_titles, m.name)]) >= COMMON_MIN_VIDEOS,
                )
                for m in v.exercises
            ),
        )
        for v in labels
    ]


def label_videos(
    videos: pd.DataFrame,
    captions: Mapping[str, Mapping[str, Any]],
    vocab: Vocabulary,
    screens: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[VideoLabel], int]:
    """라벨과, 서비스 대상이 아니어서 뺀 편수."""
    labels: list[VideoLabel] = []
    excluded = 0
    for row in videos.to_dict("records"):
        if EXCLUDED_PLAYLISTS.intersection(_split(row["playlist_ids"])):
            excluded += 1
            continue
        video_id, title = str(row["video_id"]), str(row["title"])
        screen = (screens or {}).get(video_id)
        frames = screen["frames"] if screen and "frames" in screen else []
        segments = screen_segments(frames)
        bar = screen_segments(frames, "bar")
        age_group, age_evidence = age_of(title, _split(row["playlist_titles"]))
        factors, factor_evidence = factors_of(title, segments)
        exercises = exercise_matches(
            vocab, title, str(row["description"]), captions.get(video_id), segments, bar
        )
        labels.append(
            VideoLabel(
                video_id=video_id,
                title=title,
                playlist_titles=str(row["playlist_titles"]),
                duration_sec=str(row["duration_sec"]),
                caption=str(row["caption"]),
                screen=screen_status(screen, segments),
                age_group=age_group,
                age_evidence=age_evidence,
                fitness_factors=factors,
                factor_evidence=factor_evidence,
                exercises=tuple(exercises),
            )
        )
    return mark_common(labels), excluded


def _split(cell: object) -> list[str]:
    return [x for x in str(cell).split(";") if x]


def labels_frame(labels: list[VideoLabel]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": v.video_id,
                "title": v.title,
                "playlist_titles": v.playlist_titles,
                "duration_sec": v.duration_sec,
                "age_group": v.age_group or "",
                "age_evidence": v.age_evidence,
                "fitness_factors": ";".join(v.fitness_factors),
                "factor_evidence": v.factor_evidence,
                # 본운동만. 공통 준비·마무리는 따로 적는다 (docs/dev/AI-7 §3.10)
                "exercise_names": ";".join(m.name for m in v.exercises if not m.common),
                "common_exercise_names": ";".join(m.name for m in v.exercises if m.common),
                # 영상 단위로는 비운다 — 운동별 시각은 운동 이름 → 영상 표에 (docs/dev/AI-7 §4)
                "start_sec": "",
                # 규칙이 없다 (docs/dev/AI-7 §3)
                "space": "",
                "noise": "",
                "equipment": "",
                "intensity": "",
                "caption": v.caption,
                "screen": v.screen,
                "labeler_version": LABELER_VERSION,
            }
            for v in labels
        ],
        columns=LABEL_COLUMNS,
    )


def watch_url(video_id: str, start_sec: int | None) -> str:
    """그 운동이 시작되는 지점부터 재생되는 주소. 검수·내보내기용이다.

    계약의 키는 여전히 `video_id`·`start_sec` 이고 URL 은 백엔드가 만든다 (docs/03 §4.2).
    """
    base = f"https://www.youtube.com/watch?v={video_id}"
    return base if start_sec is None else f"{base}&t={start_sec}s"


def exercises_frame(labels: list[VideoLabel]) -> pd.DataFrame:
    """운동 이름 → 영상 (docs/dev/AI-11 §4).

    이름마다 **본운동 행을 공통 준비·마무리 행보다 먼저**, 그 안에서 믿을 만한 출처부터 적는다.
    """
    rank = {s: i for i, s in enumerate(SOURCES)}
    rows = [
        {
            "exercise_name": m.name,
            "video_id": v.video_id,
            "source": m.source,
            "common": m.common,
            "evidence_text": m.evidence,
            "fitness_factors": ";".join(m.factors),
            "age_group": v.age_group or "",
            "start_sec": "" if m.start_sec is None else m.start_sec,
            "url": watch_url(v.video_id, m.start_sec),
        }
        for v in labels
        for m in v.exercises
    ]
    frame = pd.DataFrame(rows, columns=EXERCISE_COLUMNS)
    frame["_rank"] = frame["source"].map(rank)
    frame = frame.sort_values(["exercise_name", "common", "_rank", "video_id"], ignore_index=True)
    return frame.drop(columns="_rank")


def report(labels: list[VideoLabel], excluded: int, vocabulary_size: int) -> list[str]:
    """라벨 공백 리포트 (docs/dev/AI-7 §4). 0 인 칸도 0 으로 적는다."""
    ages = Counter(v.age_group for v in labels)
    factors = Counter(f for v in labels for f in v.fitness_factors)
    lines = [f"영상 {len(labels):,} · 중장년 재생목록이라 뺀 것 {excluded}", "연령대"]
    lines += [f"  {g} {ages[g]}" for g in AGE_GROUPS]
    lines.append(f"  없음 {ages[None]}")
    lines.append("요인")
    lines += [f"  {f} {factors[f]}" for f in FITNESS_FACTORS]
    lines.append(f"  없음 {sum(1 for v in labels if not v.fitness_factors)}")
    lines.append("운동명 — 가장 믿을 만한 출처 기준 · 편수 · 영상이 붙은 어휘")
    for source in SOURCES:
        videos = sum(1 for v in labels if any(m.source == source for m in v.exercises))
        names = {m.name for v in labels for m in v.exercises if m.source == source}
        lines.append(f"  {source} {videos}편 · {len(names)}/{vocabulary_size}")
    named = sum(1 for v in labels if v.exercises)
    all_names = {m.name for v in labels for m in v.exercises}
    lines.append(f"  하나라도 {named}편 · {len(all_names)}/{vocabulary_size}")
    main = sum(1 for v in labels if any(not m.common for m in v.exercises))
    common = sum(1 for v in labels for m in v.exercises if m.common)
    lines.append(f"  본운동이 하나라도 {main}편 · 공통 준비·마무리로 표시한 (운동, 영상) {common}")
    timed = sum(1 for v in labels for m in v.exercises if m.start_sec is not None)
    lines.append(f"  시각이 붙은 (운동, 영상) {timed}/{sum(len(v.exercises) for v in labels)}")
    captions = Counter(v.caption or "받지 않음" for v in labels)
    lines.append("자막 " + " · ".join(f"{k} {n}" for k, n in sorted(captions.items())))
    screens = Counter(
        "읽음" if v.screen.isdigit() else ("받을 수 없음" if v.screen else "안 읽음")
        for v in labels
    )
    lines.append("화면 " + " · ".join(f"{k} {n}" for k, n in sorted(screens.items())))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="영상에 규칙으로 라벨을 붙이고 공백을 잰다")
    ap.add_argument("--interim", default="data/interim", help="videos.csv 가 있고 라벨을 낼 곳")
    ap.add_argument("--cache", default=str(C.CACHE_DIR), help="자막·화면 캐시가 있는 곳")
    ap.add_argument("--release", default="data/release", help="처방 어휘가 있는 곳")
    args = ap.parse_args(argv)

    interim, cache = Path(args.interim), Path(args.cache)
    videos = pd.read_csv(
        interim / C.VIDEOS_FILE, encoding="utf-8-sig", dtype=str, keep_default_na=False
    )
    captions = C.JsonlCache(cache / C.CAPTION_CACHE).values()
    screens = C.JsonlCache(cache / C.SCREEN_CACHE).values()
    vocab = Vocabulary.load(Path(args.release) / VOCABULARY_FILE)
    labels, excluded = label_videos(videos, captions, vocab, screens)

    csv = {"index": False, "encoding": "utf-8-sig"}
    labels_frame(labels).to_csv(interim / LABELS_FILE, **csv)
    exercises_frame(labels).to_csv(interim / EXERCISES_FILE, **csv)
    print("\n".join(report(labels, excluded, len(vocab))))
    print(f"→ {interim / LABELS_FILE} · {interim / EXERCISES_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
