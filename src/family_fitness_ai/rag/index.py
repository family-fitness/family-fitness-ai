"""코퍼스 인덱스. data/index 를 한 번 읽어 들고 있는다.

인덱스를 만드는 일은 서비스 밖이다 — 여기서는 읽기만 한다.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from family_fitness_ai.common.copy import plain, video_title, with_subject
from family_fitness_ai.common.errors import temporarily_unavailable
from family_fitness_ai.common.settings import settings

#: 이 셋이 다 있어야 검색이 선다.
FILES = ("corpus.faiss", "corpus.json", "corpus_meta.csv")


#: 청크 본문 한 칸은 csv 기본 한도(128KB)를 넘을 수 있다. 한도는 C long 에 담기는
#: 만큼만 올린다 — Windows 는 64비트 파이썬에서도 C long 이 32비트라, 파이썬 3.11 은
#: sys.maxsize 를 넘기면 OverflowError 를 내고 코퍼스를 읽지 못했다.
_CSV_FIELD_LIMIT = min(sys.maxsize, 2**31 - 1)


def allow_long_fields() -> None:
    """corpus_meta.csv 의 긴 칸을 읽을 수 있게 csv 한도를 올린다."""
    csv.field_size_limit(_CSV_FIELD_LIMIT)


def missing_files(directory: Path | None = None) -> list[str]:
    """없는 인덱스 파일 이름. 비어 있으면 갖춰진 것이다."""
    directory = directory or settings().index_dir
    return [name for name in FILES if not (directory / name).exists()]


#: 코퍼스는 인증을 못 받은 칸을 「미달」로 적어 두었다. 원자료의 이름은 「참가」고,
#: 그 말이 화면에 나가는 말이다 — 읽는 쪽으로 옮겨 놓는다. chunk_id 는 식별자라
#: 그대로 둔다.
_REWRITE = (("미달", "참가"),)


def _readable(text: str) -> str:
    for before, after in _REWRITE:
        text = text.replace(before, after)
    return text


#: 처방표는 요인마다 등급 칸(1 · 2 · 3등급 · 참가)으로 나뉘어 있고, 코퍼스는 그 칸
#: 이름을 인용 이름과 청크 글머리에 그대로 적어 두었다(「유소년 11세 심폐지구력
#: 2등급」). 화면에 나오는 등급은 국민체력100 등급 카드(한 사람에 하나)뿐이라,
#: 요인별 등급은 읽을 때 걷어 낸다. 인덱스를 다시 만들지 않아도 된다. 등급 칸은
#: grade · chunk_id 에 남아 처방을 고를 때 그대로 쓴다.
_LABEL_GRADE = re.compile(r"^(국민체력100 운동처방 · )(.+) (\S+) (?:\d등급|참가|미달)$")
_TEXT_GRADE = re.compile(r" (\S+) (?:\d등급|참가|미달)인 ")


def _without_grade_label(source: str, label: str) -> str:
    """「… · 유소년 11세 심폐지구력 2등급」 → 「… · 심폐지구력이 비슷한 유소년 11세」."""
    if source != "prescription":
        return label
    matched = _LABEL_GRADE.match(label)
    if not matched:
        return label
    prefix, who, factor = matched.groups()
    return f"{prefix}{with_subject(factor)} 비슷한 {who}"


#: 처방 줄은 요인을 「·」 로, 영상 줄은 「;」 로 잇는다(「근력;근지구력;협응력」).
#: 「·」 로만 나누면 영상 요인이 한 덩어리 글자가 되어, 요인으로 거르는 검색에서 빠졌다.
_FACTOR_SEPARATOR = re.compile(r"[·;]")


def _factors(raw: str) -> tuple[str, ...]:
    """요인 칸을 요인 하나씩으로 나눈다. 두 구분자를 모두 받는다."""
    return tuple(f.strip() for f in _FACTOR_SEPARATOR.split(raw) if f.strip())


def _without_grade(source: str, text: str) -> str:
    """처방 청크 글머리의 「심폐지구력 2등급인」 → 「심폐지구력 수준이 비슷한」.

    글머리(「:」 앞)만 바꾼다. 뒤의 운동 목록은 자료 그대로다.
    """
    if source != "prescription":
        return text
    head, colon, rest = text.partition(":")
    return _TEXT_GRADE.sub(r" \1 수준이 비슷한 ", head, count=1) + colon + rest


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str  # criteria · prescription · video
    text: str
    citation_label: str
    citation_url: str
    age_group: str
    factors: tuple[str, ...]
    grade: str

    def label(self) -> str:
        """화면에 나가는 근거 이름.

        원자료의 「국민체력100 운동처방 · …」 가운데 점을 쉼표로 바꾼다. 유튜브 영상은
        「국민체력100 운동영상 · <유튜브 제목>」 이라 제목에서 이모지와 앞쪽 대괄호
        머리말을 걷고 「(30min)」 을 「(30분)」 으로 바꾼다. 묶을 때 쓰는 citation_label 은
        원자료 그대로 둔다.
        """
        if self.source == "video":
            head, dot, title = self.citation_label.partition(" · ")
            if dot:
                return f"{plain(head)}, {plain(video_title(title))}"
        return plain(self.citation_label)

    def citation(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "label": self.label(),
            "chunk_id": self.chunk_id,
            "url": self.citation_url or None,
        }


@dataclass(frozen=True)
class Corpus:
    index: faiss.Index
    chunks: tuple[Chunk, ...]
    meta: dict[str, object]

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        query = np.asarray([vector], dtype="float32")
        scores, ids = self.index.search(query, min(k, len(self.chunks)))
        out = []
        for chunk_id, score in zip(ids[0], scores[0], strict=True):
            if chunk_id < 0:
                continue
            out.append((self.chunks[int(chunk_id)], float(score)))
        return out

    def by_id(self, chunk_id: str) -> Chunk | None:
        return self._lookup.get(chunk_id)

    @property
    def _lookup(self) -> dict[str, Chunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}


@lru_cache
def corpus() -> Corpus:
    directory = settings().index_dir
    # 먼저 있는지 본다. 없는 채로 faiss 에 넘기면 C++ 쪽 오류가 그대로 올라와
    # 서버 안 경로까지 응답에 실린다.
    absent = missing_files(directory)
    if absent:
        raise temporarily_unavailable(
            f"코퍼스 인덱스가 없습니다 ({', '.join(absent)}). 배포에 data/index 를 실었는지 보세요"
        )
    index = faiss.read_index(str(directory / "corpus.faiss"))
    meta = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))

    allow_long_fields()
    chunks = []
    with (directory / "corpus_meta.csv").open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            factors = _factors(row.get("fitness_factors") or "")
            source = row["source"]
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    source=source,
                    text=_without_grade(source, _readable(row["text"])),
                    citation_label=_without_grade_label(source, _readable(row["citation_label"])),
                    citation_url=row.get("citation_url") or "",
                    age_group=row.get("age_group") or "",
                    factors=factors,
                    grade=_readable(row.get("grade") or ""),
                )
            )

    if index.ntotal != len(chunks):
        raise RuntimeError(f"인덱스와 메타의 줄 수가 다르다: {index.ntotal} vs {len(chunks)}")
    return Corpus(index=index, chunks=tuple(chunks), meta=meta)
