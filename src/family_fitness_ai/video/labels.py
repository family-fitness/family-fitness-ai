"""클립 이름에 뜻을 붙인다.

    python -m family_fitness_ai.video.labels [--llm]

영상 속 이름과 처방 어휘는 같은 말을 쓰지 않는다. 유아기 영상은 「공을 던져
붙여요」처럼 놀이 이름으로 부르고, 처방문은 「팔벌려뛰기」처럼 동작 이름으로
적는다. 그 사이를 잇는 표가 data/release/clip_labels.csv 다.

이어 붙이는 방법을 셋으로 나눈 것은 재 보고 정한 것이다.

    exact   글자가 같다. 유아기 놀이 이름은 처방 어휘에도 그대로 있다.
    embed   뜻이 같다고 볼 만큼 가깝다(0.90 이상).
    llm     그 아래는 임베딩이 틀린 짝을 자주 골랐다 — 「캐치볼을 해요」에
            「낚시를 해요」를 붙이는 식이다. 그래서 0.90 아래는 기계 점수로
            정하지 않고, 후보만 추려 LLM 에게 고르게 하거나 비워 둔다.

사람이 고친 줄(source=human)은 다시 돌려도 그대로 둔다.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from family_fitness_ai.common import llm
from family_fitness_ai.common.items import FACTORS
from family_fitness_ai.common.settings import settings
from family_fitness_ai.rag.embed import embed
from family_fitness_ai.rag.index import corpus
from family_fitness_ai.video import vocabulary

#: 이 위로는 임베딩 점수만 믿는다. 아래는 LLM 이 고르거나 비운다.
EMBED_TRUST = 0.90
#: LLM 에게 보여 줄 후보 수.
CANDIDATES = 8

PHASES = ("준비운동", "본운동", "정리운동")

#: 운동이 아닌 칸. 쉬는 시간과 인사는 미션에 넣지 않는다.
_NOT_EXERCISE = ("휴식", "쉬는", "인사", "안내", "소개", "마무리 인사", "오늘의")
#: 뛰는 동작은 아랫집에 울린다.
_LOUD = ("뛰기", "뛰어", "점프", "달리기", "줄넘기", "콩콩", "후다닥", "지그재그")
#: 도구가 있어야 하는 동작.
_PROPS = (
    "밴드",
    "볼",
    "공",
    "덤벨",
    "의자",
    "박스",
    "스틱",
    "매트",
    "풍선",
    "훌라",
    "빌리보",
    "뽁뽁이",
    "줄넘기",
    "사다리",
    "책상",
    "낚시",
    "바벨",
    "케틀벨",
)
#: 집에서 하기 어려운 것.
_OUTDOOR = ("수영", "자전거", "트레드밀", "사다리", "계단", "공원", "달리기")
_WARMUP = ("스트레칭", "체조", "늘리기", "늘려", "풀기", "돌리기", "준비")


@dataclass
class Label:
    name_on_video: str
    exercise_name: str
    fitness_factor: str
    phase: str
    is_exercise: bool
    home_ok: bool
    quiet: bool
    needs_props: bool
    source: str
    score: float


def _rules(name: str, phase_hint: str) -> dict[str, object]:
    lowered = name.replace(" ", "")
    is_exercise = not any(word.replace(" ", "") in lowered for word in _NOT_EXERCISE)
    quiet = not any(word in lowered for word in _LOUD)
    needs_props = any(word in lowered for word in _PROPS)
    home_ok = not any(word in lowered for word in _OUTDOOR)
    phase = phase_hint
    if not phase and any(word in lowered for word in _WARMUP):
        phase = "준비운동"
    return {
        "phase": phase or "본운동",
        "is_exercise": is_exercise,
        "home_ok": home_ok,
        "quiet": quiet,
        "needs_props": needs_props,
    }


def _clip_names(release_dir: Path) -> dict[str, tuple[str, str]]:
    """클립 이름 → (화면에 뜬 단계, 연령대)."""
    ages = {
        chunk.chunk_id.split(":", 1)[1]: chunk.age_group
        for chunk in corpus().chunks
        if chunk.source == "video"
    }
    out: dict[str, tuple[str, str]] = {}
    with (release_dir / "video_clips.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["name_on_video"]
            phase = row["phase_on_video"]
            age_group = ages.get(row["video_id"], "")
            before = out.get(name)
            if before is None or (not before[0] and phase):
                out[name] = (phase, age_group)
    return out


_SYSTEM = """너는 국민체력100 운동영상의 화면 이름을 처방 어휘에 잇는 일을 한다.

- 후보 중에 **같은 동작**이 있을 때만 고른다. 비슷해 보인다고 고르지 않는다.
  「캐치볼을 해요」와 「낚시를 해요」는 다른 동작이다. 없으면 빈 문자열이다.
- fitness_factor 는 여덟 중 하나이거나 빈 문자열이다:
  심폐지구력·근력·근지구력·유연성·민첩성·순발력·협응력·평형성
- phase 는 준비운동·본운동·정리운동 중 하나다. 늘이고 푸는 동작은 준비운동이나
  정리운동, 힘을 쓰거나 숨이 차는 동작은 본운동이다.
- is_exercise 는 그것이 운동 동작일 때만 참이다. 휴식·인사·안내는 거짓이다.
- quiet 는 아랫집에 울리지 않을 때 참이다(뛰는 동작은 거짓).
- home_ok 는 좁은 집 안에서 할 수 있을 때 참이다.
- needs_props 는 도구가 있어야 할 때 참이다."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name_on_video": {"type": "string"},
                    "exercise_name": {"type": "string"},
                    "fitness_factor": {"type": "string"},
                    "phase": {"type": "string", "enum": list(PHASES)},
                    "is_exercise": {"type": "boolean"},
                    "home_ok": {"type": "boolean"},
                    "quiet": {"type": "boolean"},
                    "needs_props": {"type": "boolean"},
                },
                "required": [
                    "name_on_video",
                    "exercise_name",
                    "fitness_factor",
                    "phase",
                    "is_exercise",
                    "home_ok",
                    "quiet",
                    "needs_props",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def _ask_llm(
    batch: list[dict[str, object]], backend: str = "", model: str = ""
) -> dict[str, dict[str, object]]:
    payload = llm.ask_json_or_none(
        json.dumps({"묶음": batch}, ensure_ascii=False),
        _SCHEMA,
        _SYSTEM,
        max_tokens=8000,
        backend=backend,
        model=model,
    )
    if not payload:
        return {}
    out = {}
    for row in payload.get("labels", []):
        name = str(row.get("name_on_video", ""))
        picked = str(row.get("exercise_name", "")).strip()
        if picked and picked not in vocabulary.all_names():
            picked = ""  # 어휘에 없는 이름을 지어냈다면 버린다
        factor = str(row.get("fitness_factor", "")).strip()
        out[name] = {
            "exercise_name": picked,
            "fitness_factor": factor if factor in FACTORS else "",
            "phase": row.get("phase") or "본운동",
            "is_exercise": bool(row.get("is_exercise", True)),
            "home_ok": bool(row.get("home_ok", True)),
            "quiet": bool(row.get("quiet", True)),
            "needs_props": bool(row.get("needs_props", False)),
        }
    return out


def build(
    release_dir: Path,
    *,
    use_llm: bool,
    batch_size: int = 20,
    backend: str = "",
    model: str = "",
    refresh: bool = False,
) -> list[Label]:
    """이미 붙여 둔 이름은 그대로 두고 새 이름만 붙인다.

    사람이 고친 줄(source=human)은 무엇을 해도 지키고, 기계가 붙인 줄도 기본으로는
    다시 묻지 않는다 — 클립을 다시 끊을 때마다 LLM 을 처음부터 다시 부르면
    돈만 나가고 답은 그대로다. 전부 다시 매기려면 refresh 를 준다.
    """
    kept: dict[str, Label] = {}
    path = release_dir / "clip_labels.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("source") == "human" or not refresh:
                    kept[row["name_on_video"]] = Label(
                        name_on_video=row["name_on_video"],
                        exercise_name=row["exercise_name"],
                        fitness_factor=row["fitness_factor"],
                        phase=row["phase"],
                        is_exercise=row["is_exercise"] == "True",
                        home_ok=row["home_ok"] == "True",
                        quiet=row["quiet"] == "True",
                        needs_props=row["needs_props"] == "True",
                        source=row.get("source", "rule"),
                        score=float(row["score"] or 0),
                    )

    found = _clip_names(release_dir)
    todo = [name for name in sorted(found) if name not in kept]
    vocab = list(vocabulary.all_names())
    lookup = {vocabulary.normalized(name): name for name in vocab}

    matrix = embed(vocab)
    queries = embed(todo) if todo else np.zeros((0, matrix.shape[1]), dtype="float32")
    similarity = queries @ matrix.T if len(todo) else np.zeros((0, len(vocab)))

    labels: list[Label] = list(kept.values())
    for_llm: list[tuple[str, dict[str, object]]] = []

    for index, name in enumerate(todo):
        phase_hint, age_group = found[name]
        base = _rules(name, phase_hint)
        exact = lookup.get(vocabulary.normalized(name))
        if exact:
            labels.append(
                Label(
                    name,
                    exact,
                    "",
                    str(base["phase"]),
                    bool(base["is_exercise"]),
                    bool(base["home_ok"]),
                    bool(base["quiet"]),
                    bool(base["needs_props"]),
                    "exact",
                    1.0,
                )
            )
            continue

        order = np.argsort(-similarity[index])
        top = float(similarity[index][order[0]]) if len(vocab) else 0.0
        if top >= EMBED_TRUST:
            labels.append(
                Label(
                    name,
                    vocab[order[0]],
                    "",
                    str(base["phase"]),
                    bool(base["is_exercise"]),
                    bool(base["home_ok"]),
                    bool(base["quiet"]),
                    bool(base["needs_props"]),
                    "embed",
                    round(top, 4),
                )
            )
            continue

        allowed = set(vocabulary.by_age_group().get(age_group, ())) or set(vocab)
        candidates = [vocab[i] for i in order if vocab[i] in allowed][:CANDIDATES]
        for_llm.append(
            (
                name,
                {
                    "name_on_video": name,
                    "연령대": age_group,
                    "화면_단계": phase_hint,
                    "후보": candidates,
                },
            )
        )

    answers: dict[str, dict[str, object]] = {}
    if use_llm and for_llm and llm.available():
        for start in range(0, len(for_llm), batch_size):
            batch = [payload for _, payload in for_llm[start : start + batch_size]]
            answers.update(_ask_llm(batch, backend, model))
            print(f"   LLM {min(start + batch_size, len(for_llm))}/{len(for_llm)}")

    for name, _ in for_llm:
        phase_hint, _age = found[name]
        base = _rules(name, phase_hint)
        answer = answers.get(name)
        if answer:
            labels.append(
                Label(
                    name,
                    str(answer["exercise_name"]),
                    str(answer["fitness_factor"]),
                    str(answer["phase"]),
                    bool(answer["is_exercise"]),
                    bool(answer["home_ok"]),
                    bool(answer["quiet"]),
                    bool(answer["needs_props"]),
                    "llm",
                    0.0,
                )
            )
        else:
            labels.append(
                Label(
                    name,
                    "",
                    "",
                    str(base["phase"]),
                    bool(base["is_exercise"]),
                    bool(base["home_ok"]),
                    bool(base["quiet"]),
                    bool(base["needs_props"]),
                    "rule",
                    0.0,
                )
            )

    labels.sort(key=lambda label: label.name_on_video)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(labels[0]).keys()))
        writer.writeheader()
        for label in labels:
            writer.writerow(asdict(label))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", default=str(settings().release_dir))
    parser.add_argument("--llm", action="store_true", help="임베딩이 못 정한 것을 LLM 이 고른다")
    parser.add_argument("--backend", default="", help="claude | gemini (기본은 .env)")
    parser.add_argument("--model", default="")
    parser.add_argument("--refresh", action="store_true", help="붙여 둔 이름까지 다시 매긴다")
    args = parser.parse_args()

    labels = build(
        Path(args.release_dir),
        use_llm=args.llm,
        backend=args.backend,
        model=args.model,
        refresh=args.refresh,
    )
    counts: dict[str, int] = {}
    for label in labels:
        counts[label.source] = counts.get(label.source, 0) + 1
    mapped = sum(1 for label in labels if label.exercise_name)
    print(f"이름 {len(labels)}개 · 처방 어휘에 이어짐 {mapped}개 · 출처별 {counts}")


if __name__ == "__main__":
    main()
