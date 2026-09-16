"""운동 표 — 미션을 대표하는 운동 하나가 한 행이다 (docs/05 §3.1).

**합치는 근거는 「같은 처방 칸에 함께 나오지 않는다」다.** 공단이 두 이름을 한 칸에
나란히 처방했다면 그것은 별개 운동이다 — 이름이 얼마나 닮았는지는 근거가 아니다.
반대 방향(함께 안 나오면 같다)은 쓸 수 없다: 칸의 운동이 청크에서 잘리기 전이라도
한 칸에 모든 운동이 들어가지는 않는다.

그 잣대로 재니 **문자열로 합칠 것이 하나도 없다** — 숫자·로마숫자 꼬리 10묶음,
괄호 2묶음이 전부 동시 출현으로 막힌다 (`하지 루틴 스트레칭1` + `2`,
`목 굽힘/ 폄 I`·`II`·`III`, `대퇴이두근 스트레칭` + `넙다리 뒤쪽 스트레칭`).
표기 차이 4건은 `identity` 가 이미 합쳐 어휘가 641개다.

**그래서 이 표의 일은 합치는 것이 아니라 가르는 것이다.** 이름이 운동 하나를
가리키지 못하면(`맨몸운동  루틴프로그램`) 미션 제목이 될 수 없다 — `kind` 로 표시한다.

실행:
    python -m family_fitness_ai.mission.exercises
"""

from __future__ import annotations

import argparse
import itertools
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..common.settings import RELEASE_DIR
from ..rag.prescription import VOCABULARY_FILE, identity
from .cells import CELLS_FILE

EXERCISES_FILE = "exercises.csv"

COLUMNS = [
    "exercise_id",
    "exercise_name",
    "raw_forms",
    "merge_reason",
    "kind",
    "phase_share",
    "prescribed",
    "age_groups",
    "blocked_with",
    "indoor",
]

# 운동 하나가 아니라 여러 운동의 묶음을 가리키는 이름. **미션 제목이 될 수 없다** —
# 「맨몸운동 루틴프로그램 15분」을 받은 사람은 무엇을 할지 알 수 없다.
# 빈도가 가장 높은 이름들이라(처방의 24.8%) 순위로 고르면 상단을 차지한다.
BUNDLE = re.compile(r"루틴|프로그램|세트|체조")

KIND_SINGLE = "낱개"
KIND_BUNDLE = "묶음"

# 합침 후보를 만드는 규칙. **후보일 뿐이다** — 동시 출현이 하나라도 있으면 막힌다.
_SUFFIX = re.compile(r"\s*(?:[0-9]+|[IVXivx]+)\s*$")
_PAREN = re.compile(r"\s*\([^)]*\)\s*")

# `indoor` 는 비운다. 이름으로 판정했다는 옛 수치(14개/497개)는 판정 규칙도
# 스크립트도 없고 14+497 이 641 과 맞지 않는다 — 근거로 쓰지 않는다 (docs/05 §6).
NOT_FILLED = ("indoor",)


def _norm(name: str) -> str:
    return re.sub(r"[\s·ㆍ]", "", unicodedata.normalize("NFKC", name))


def merge_keys(name: str) -> dict[str, str]:
    """후보를 묶는 키들. 값이 같은 이름끼리가 한 후보 묶음이다."""
    flat = _norm(name)
    return {
        "표기": flat,
        "꼬리": _SUFFIX.sub("", flat),
        "괄호": _norm(_PAREN.sub("", name)),
    }


def co_occurring(cells: pd.DataFrame) -> set[tuple[str, str]]:
    """같은 칸에 함께 처방된 이름 쌍. **이 쌍은 합치지 않는다.**"""
    pairs: set[tuple[str, str]] = set()
    for listed in cells["exercise_names"].fillna(""):
        names = sorted({x for x in str(listed).split(";") if x})
        pairs.update(itertools.combinations(names, 2))
    return pairs


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class Candidate:
    """합침 후보 묶음 하나와 그 판정."""

    rule: str
    names: tuple[str, ...]
    blocked: tuple[tuple[str, str], ...]

    @property
    def merged(self) -> bool:
        return not self.blocked


def candidates(names: list[str], together: set[tuple[str, str]]) -> list[Candidate]:
    """이름이 둘 이상 모인 후보 묶음을 규칙마다 찾고, 동시 출현으로 판정한다."""
    found: list[Candidate] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for rule in ("표기", "꼬리", "괄호"):
        groups: dict[str, list[str]] = defaultdict(list)
        for name in names:
            groups[merge_keys(name)[rule]].append(name)
        for group in groups.values():
            if len(group) < 2:
                continue
            key = (rule, tuple(sorted(group)))
            if key in seen:
                continue
            seen.add(key)
            blocked = tuple(
                _pair(a, b)
                for a, b in itertools.combinations(sorted(group), 2)
                if _pair(a, b) in together
            )
            found.append(Candidate(rule, tuple(sorted(group)), blocked))
    return found


def phase_shares(cells: pd.DataFrame) -> dict[str, dict[str, float]]:
    """운동마다 준비·본·정리의 처방 비중. 칸 표의 단계별 횟수를 더해서 낸다."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in cells.iterrows():
        phase = str(row["phase"])
        names = [x for x in str(row["exercise_names"] or "").split(";") if x]
        counts = [x for x in str(row["exercise_counts"] or "").split(";") if x]
        for name, count in zip(names, counts, strict=False):
            totals[identity(name)][phase] += float(count)
    shares: dict[str, dict[str, float]] = {}
    for key, by_phase in totals.items():
        total = sum(by_phase.values())
        if total:
            shares[key] = {p: n / total for p, n in by_phase.items()}
    return shares


def _format_share(share: dict[str, float]) -> str:
    order = ("준비운동", "본운동", "정리운동")
    return ";".join(f"{p}:{share[p]:.3f}" for p in order if p in share)


def build(vocabulary: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, list[Candidate]]:
    """운동 표와 후보 판정 내역을 낸다."""
    names = vocabulary["exercise_name"].astype(str).tolist()
    together = co_occurring(cells)
    found = candidates(names, together)
    shares = phase_shares(cells)

    # 막히지 않은 후보만 대표 이름으로 접는다 (가장 많이 처방된 원문이 대표다)
    representative: dict[str, str] = {}
    reason: dict[str, str] = {}
    counts = dict(zip(names, vocabulary["count"].astype(int), strict=False))
    for candidate in found:
        if not candidate.merged:
            continue
        head = max(candidate.names, key=lambda n: counts.get(n, 0))
        for name in candidate.names:
            if name != head:
                representative[name] = head
                reason[head] = candidate.rule

    blocked_with: dict[str, set[str]] = defaultdict(set)
    for candidate in found:
        for a, b in candidate.blocked:
            blocked_with[a].add(b)
            blocked_with[b].add(a)

    rows = []
    for _, row in vocabulary.iterrows():
        name = str(row["exercise_name"])
        if name in representative:  # 대표에게 접혔다 — 행을 내지 않는다
            continue
        key = identity(name)
        rows.append(
            {
                "exercise_id": key,
                "exercise_name": name,
                "raw_forms": str(row["raw_forms"]),
                "merge_reason": reason.get(name, ""),
                "kind": KIND_BUNDLE if BUNDLE.search(name) else KIND_SINGLE,
                "phase_share": _format_share(shares.get(key, {})),
                "prescribed": int(row["count"]),
                "age_groups": str(row["age_groups"]),
                "blocked_with": ";".join(sorted(blocked_with.get(name, ()))),
                "indoor": "",
            }
        )
    frame = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        "prescribed", ascending=False, ignore_index=True
    )
    return frame, found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="운동 표를 낸다 (미션의 운동 축)")
    ap.add_argument("--release", default=str(RELEASE_DIR), help="낼 곳 (커밋한다)")
    args = ap.parse_args(argv)

    release = Path(args.release)
    vocabulary_path, cells_path = release / VOCABULARY_FILE, release / CELLS_FILE
    for path in (vocabulary_path, cells_path):
        if not path.exists():
            print(f"[중단] 없다: {path}")
            return 1

    vocabulary = pd.read_csv(vocabulary_path, encoding="utf-8-sig")
    cells = pd.read_csv(cells_path, encoding="utf-8-sig")
    frame, found = build(vocabulary, cells)
    frame.to_csv(release / EXERCISES_FILE, index=False, encoding="utf-8-sig")

    merged = [c for c in found if c.merged]
    blocked = [c for c in found if not c.merged]
    kinds = frame["kind"].value_counts()
    print(f"운동 {len(frame)}행 (어휘 {len(vocabulary)} → 합친 것 {len(vocabulary) - len(frame)})")
    single, bundle = kinds.get(KIND_SINGLE, 0), kinds.get(KIND_BUNDLE, 0)
    print(f"  {KIND_SINGLE} {single} · {KIND_BUNDLE} {bundle}")
    print(f"후보 {len(found)}묶음 — 합쳤다 {len(merged)} · 동시 출현으로 막혔다 {len(blocked)}")
    for candidate in blocked:
        print(f"  [막힘·{candidate.rule}] {' | '.join(candidate.names)}")
    print(f"채우지 않은 열: {', '.join(NOT_FILLED)}")
    print(f"→ {release / EXERCISES_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
