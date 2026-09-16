#!/usr/bin/env python3
"""백엔드 와이어의 필드 이름과 우리 pydantic 별칭을 기계로 맞춘다.

`AiWire.kt` 의 `@JsonProperty("...")` 이름(없으면 코틀린 프로퍼티 이름)이 정본이다.
사람이 눈으로 비교하면 한 글자를 놓친다.

    python scripts/wire_check.py
    python scripts/wire_check.py --backend ~/temp/family-fitness-be
    FAMILY_FITNESS_BE=~/temp/family-fitness-be python scripts/wire_check.py

백엔드 저장소가 없으면 **skip 으로 끝난다** (exit 0). CI 에는 백엔드가 없다.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from family_fitness_ai.api.coach_schemas import WIRE_CLASSES, wire_names  # noqa: E402

WIRE_PATH = "backend/src/main/kotlin/kr/ac/kookmin/familyfitness/shared/ai/AiWire.kt"
DEFAULT_BACKEND = "~/temp/family-fitness-be"

CLASS_RE = re.compile(r"data class (\w+)\s*\(")
JSONPROP_RE = re.compile(r'@JsonProperty\("([^"]+)"\)')
VAL_RE = re.compile(r"\bval\s+(\w+)\s*:")


def param_list(text: str, open_paren: int) -> str:
    """`(` 부터 짝이 맞는 `)` 까지. 클래스 본문의 중첩 클래스는 들어오지 않는다."""
    depth = 0
    for i in range(open_paren, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : i]
    return ""


def parse_wire(text: str) -> dict[str, list[str]]:
    """코틀린 data class 이름 → 와이어에 나가는 필드 이름들."""
    parsed: dict[str, list[str]] = {}
    for match in CLASS_RE.finditer(text):
        body = param_list(text, match.end() - 1)
        fields: list[str] = []
        pending: str | None = None
        for line in body.splitlines():
            prop = JSONPROP_RE.search(line)
            if prop:
                pending = prop.group(1)
            name = VAL_RE.search(line)
            if name:
                fields.append(pending or name.group(1))
                pending = None
        parsed[match.group(1)] = fields
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default=os.environ.get("FAMILY_FITNESS_BE", DEFAULT_BACKEND),
        help=f"백엔드 저장소 경로 (기본 {DEFAULT_BACKEND})",
    )
    args = parser.parse_args()

    wire_file = pathlib.Path(args.backend).expanduser() / WIRE_PATH
    if not wire_file.exists():
        print(f"skip: 백엔드 와이어가 없다 — {wire_file}")
        return 0

    parsed = parse_wire(wire_file.read_text(encoding="utf-8"))
    missing_total = 0
    extra_total = 0
    unseen: list[str] = []

    for kotlin_name, model in WIRE_CLASSES.items():
        if kotlin_name not in parsed:
            unseen.append(kotlin_name)
            continue
        theirs = set(parsed[kotlin_name])
        ours = wire_names(model)
        missing = sorted(theirs - ours)
        extra = sorted(ours - theirs)
        missing_total += len(missing)
        extra_total += len(extra)
        mark = "ok  " if not (missing or extra) else "DIFF"
        line = f"{mark} {kotlin_name:24s} → {model.__name__:20s} {len(ours):2d}개"
        if missing:
            line += f" · 우리에게 없음 {missing}"
        if extra:
            line += f" · 와이어에 없음 {extra}"
        print(line)

    if unseen:
        print(f"DIFF 와이어에서 찾지 못한 클래스: {sorted(unseen)}")

    print(f"\n누락 {missing_total} · 여분 {extra_total} · 미발견 클래스 {len(unseen)}")
    return 0 if not (missing_total or extra_total or unseen) else 1


if __name__ == "__main__":
    sys.exit(main())
