#!/usr/bin/env bash
# 아이·성인 × 입력 수준(L0·L1·L2) 확인용 요청.
#
#   ./scripts/check_levels.sh              # 요약만
#   RAW=1 ./scripts/check_levels.sh        # 응답 원문까지
#   BASE=http://127.0.0.1:8001/v1 ./scripts/check_levels.sh
#
# API 서버(:8000)가 떠 있어야 한다(임베딩은 그 안에서 돈다). 편성은 LLM 이 켜져 있으면
# 한 번에 한 번씩 부른다 — 끄려면 서버를 COACH_LLM=0 으로 띄운다.
set -euo pipefail
BASE=${BASE:-http://127.0.0.1:8000/v1}
H='Content-Type: application/json'

# ── 입력 ──────────────────────────────────────────────────────────────────
# L0 나이·성별 / L1 + 키·몸무게 / L2 + 측정값
# L2 는 그 연령대 기준항목을 전부 채웠다. 하나라도 빠지면 등급은 null 이다.
CHILD='"age":11,"age_unit":"세","sex":"F"'
CHILD_BODY='"height_cm":148.0,"weight_kg":41.0'
CHILD_MEAS='"measurements":{"020":70,"028":41.3,"009":30,"012":4.0,"043":28,"022":133,"044":8,"004":60.0}'

ADULT='"age":41,"age_unit":"세","sex":"F"'
ADULT_BODY='"height_cm":162.0,"weight_kg":57.0'
ADULT_MEAS='"measurements":{"020":20,"035":31.0,"028":40.0,"019":20,"012":12.0,"021":14.5,"040":0.42,"022":140,"041":0.40,"003":28.0}'

show() { if [ "${RAW:-}" = 1 ]; then jq .; else jq -c "$1"; fi; }

assess() {  # $1 이름 $2 본문
  printf '\n── 평가 · %s\n' "$1"
  curl -s -X POST "$BASE/fitness/assessment" -H "$H" -d "$2" |
    show '{input_level, age_group, focus: .child_scope.focus_one.factor, grade: .parent_scope.grade,
           factors: [.parent_scope.factors[] | "\(.factor) \(.percentile)"], low_sample}'
}

run() {  # $1 이름 $2 본문
  printf '\n── 편성 · %s\n' "$1"
  local id state
  id=$(curl -s -X POST "$BASE/coach/runs" -H "$H" -d "$2" | jq -r .run_id)
  until state=$(curl -s "$BASE/coach/runs/$id"); [ "$(jq -r .status <<<"$state")" != running ]; do sleep 1.5; done
  show '{status, refusal_reason, steps: [.steps[] | "\(.name) \(.status) — \(.summary)"],
         notices: .proposal.notices, days: (.proposal.missions | length),
         first_day: (.proposal.missions[0] | {title, duration_min, clips: (.sessions | length)})}' <<<"$state"
}

profile() { printf '{"profile_ref":"p",%s}' "$1"; }
mover()   { printf '{"profile_refs":[{"ref":"p","role":"주행자","input_level":"%s",%s}],"period":{"start_date":"2026-09-21","weeks":1},"constraints":{"days_per_week":3,"minutes_per_session":15}}' "$1" "$2"; }

# ── 체력 평가 ─────────────────────────────────────────────────────────────
assess "아이 L0" "$(profile "$CHILD")"
assess "아이 L1" "$(profile "$CHILD,$CHILD_BODY")"
assess "아이 L2" "$(profile "$CHILD,$CHILD_BODY,$CHILD_MEAS")"
assess "성인 L0" "$(profile "$ADULT")"
assess "성인 L1" "$(profile "$ADULT,$ADULT_BODY")"
assess "성인 L2" "$(profile "$ADULT,$ADULT_BODY,$ADULT_MEAS")"

# ── 한 주 편성 ────────────────────────────────────────────────────────────
run "아이 L0" "$(mover L0 "$CHILD")"
run "아이 L1" "$(mover L1 "$CHILD,$CHILD_BODY")"
run "아이 L2" "$(mover L2 "$CHILD,$CHILD_BODY,$CHILD_MEAS")"
run "성인 L0" "$(mover L0 "$ADULT")"
run "성인 L1" "$(mover L1 "$ADULT,$ADULT_BODY")"
run "성인 L2" "$(mover L2 "$ADULT,$ADULT_BODY,$ADULT_MEAS")"
