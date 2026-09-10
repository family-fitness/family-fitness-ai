# dev · 구현 순서

`01`~`04` 가 **무엇을 만드는가**를 정한다. 이 디렉터리는 **그것을 어떤 순서로,
무엇을 산출하며 만드는가**를 적는다. 한 파일이 한 이슈다.

규약을 여기에 복사하지 않는다. 규약이 필요하면 `../0N` 을 가리킨다.

---

## 1. 지금 어디까지 왔나

| 구역 | 상태 |
|---|---|
| `docs/01`~`04` | 확정. 백엔드 실물과 대조해 갱신했다 — [AI-13](AI-13-backend-contract-reconciliation.md) |
| `stats/` · `ingest/` | **동작한다.** 원자료 → 점수 눈금 → 산출물 6종 → 채점 CLI |
| `api/` · `common/` | **동작한다.** 설정·오류·로그·`/healthz`·`/readyz`·`/v1/fitness/assessment` |
| `graph/` · `rag/` · `labeling/` | `__init__.py` 만 있다 — **여기가 남은 일이다** |
| CI | `ruff` · `mypy` · `pytest` |

**백엔드는 우리 다섯 경로를 부를 준비가 끝났다.** `AiGateway` 가 타임아웃·재시도까지
`../03` §2.3 그대로 구현돼 있고, 지금은 스텁이 가짜 응답을 낸다.

**그런데 실제 호출부는 셋뿐이다.**

| 우리 엔드포인트 | 게이트웨이 | 호출부 |
|---|---|---|
| `/coach/runs` · `/coach/runs/{id}` | ✅ | ✅ `CoachRunExecutor` |
| `/coach/messages` | ✅ | ✅ `CoachChatService` |
| `/fitness/trajectory` | ✅ | ✅ `PredictionService` |
| `/fitness/assessment` | ✅ | **없다** — 백엔드가 백분위·등급을 자체 계산한다 |
| `/videos/search` | ✅ | **없다** — 백엔드가 `exercise_videos` 를 직접 조회한다 |

**그래서 목표를 coach 로 옮긴다** (§3).

---

## 2. 순서와 의존

```
AI-1 분포 ✅ ─ AI-2 등급 ✅ ─┐
                             ├─ AI-4 assessment ✅ (호출부 없음 · 보류)
AI-3 서비스 골격 ✅ ─────────┘
                    │
AI-6 처방 코퍼스 ───┤
                    ├─ AI-8 임베딩·색인·검색 ─┬─ AI-10 coach/messages
AI-7 영상 라벨링 ───┘                          └─ AI-11 coach/runs
                                                      │
                                                AI-12 배포·관측
```

| 이슈 | 무엇 | 선행 | 상태 |
|---|---|---|---|
| [AI-13](AI-13-backend-contract-reconciliation.md) | 백엔드 대조 | — | ✅ 갱신됨 |
| [AI-1](AI-1-age-band-distribution.md) | 연령 구간별 점수 분포 | — | ✅ |
| [AI-2](AI-2-grade-card.md) | 등급 판정과 또래 등급 분포 | AI-1 | ✅ |
| [AI-3](AI-3-service-skeleton.md) | FastAPI 골격 | — | ✅ |
| [AI-4](AI-4-assessment-endpoint.md) | `POST /fitness/assessment` | AI-2·3 | ✅ **보류** — 호출부 없음 |
| **[AI-6](AI-6-prescription-corpus.md)** | 처방 어휘·청크 | AI-1 | **다음** |
| **[AI-8](AI-8-vector-index.md)** | 임베딩·색인·검색 | AI-6 | 막는 것 없음 |
| **[AI-10](AI-10-coach-messages.md)** | `POST /coach/messages` | AI-8 | 호출부 있음 |
| **[AI-11](AI-11-coach-runs.md)** | 미션 편성 그래프 | AI-4·10 | 호출부·저장 자리 있음 |
| [AI-7](AI-7-video-labeling.md) | 영상 수집·라벨링 | — | 검색 품질용 |
| [AI-5](AI-5-trajectory-bands.md) | `POST /fitness/trajectory` | AI-3 | 호출부 있으나 스텁으로 화면이 돈다 |
| [AI-9](AI-9-videos-search.md) | `POST /videos/search` | AI-8 | 백엔드 자체 조회로 충분 |
| [AI-12](AI-12-deploy.md) | 컨테이너·배포·관측 | AI-3 | 마지막 |

---

## 3. 묶음 — coach 를 먼저 세운다

**우리가 대체 불가능한 것은 근거다.** 백분위·밴드·영상 조회는 백엔드가 이미 하고
있고 그쪽이 데이터를 갖고 있으니 그게 맞다. 우리만 할 수 있는 것은 **인용을 붙인
편성과 답변**이다 — 그래서 목표를 coach 로 옮긴다.

| 묶음 | 이슈 | 끝나면 보여줄 수 있는 것 |
|---|---|---|
| **M1 · 근거가 생긴다** | AI-6 · AI-8 | 처방 청크가 색인되고 연령 필터가 걸린 검색이 돈다 |
| **M2 · 코치가 답한다** | AI-10 | 질문에 공단 자료를 인용해 답하고, 없으면 거부한다 |
| **M3 · 미션을 편성한다** | AI-11 | 주간 제안이 인용과 함께 나오고 백엔드가 승인 게이트를 태운다 |
| **M4 · 품질과 배포** | AI-7 · AI-12 | 영상 라벨로 검색이 좋아지고 두 컨테이너가 뜬다 |

**보류** — AI-4(완료·호출부 없음) · AI-5 · AI-9. 계약에는 남는다. 소비자가 생기면
그대로 동작한다.

**M1 은 아무것도 기다리지 않는다.** `ai_documents` 가 사라져 벡터 저장소가 우리
것이 됐고 ([AI-13](AI-13-backend-contract-reconciliation.md) §3.3), 임베딩 모델도
우리가 고른다.

---

## 4. 지금 답이 필요한 것

미결의 정본은 각 문서다. 여기서는 **무엇이 무엇을 막고 있는지만** 모은다.

| # | 답이 필요한 것 | 정본 | 막는 것 |
|---|---|---|---|
| ① | `fitness_test_items` 에 `score` 컬럼이 없다 | [AI-13](AI-13-backend-contract-reconciliation.md) §3.5 | 저장 — 구현은 안 막는다 |
| ② | 영상 라벨을 백엔드에 넣을 경로 | [AI-13](AI-13-backend-contract-reconciliation.md) §2.2 | AI-7 적재 |
| ③ | `SIM_THRESHOLD` 값 | `04` §6 ② | AI-8 이 재는 것이 산출이다 |
| ④ | 임베딩 모델과 차원 | `01` §7 ② · `04` §6 ① | **우리가 고른다.** AI-8 착수 전에 정한다 |

**막는 것이 하나도 없다.** ①②는 저장·적재 쪽이고 구현을 멈추지 않는다. ③은 AI-8 이
스스로 재고, ④는 우리 결정이다.

### 4.1 등급은 백엔드를 따른다

`Grade.ofPercentile(90/75/50)` 로 백엔드가 항목마다 등급을 붙인다. 우리 준거참조
판정(`02` §5.2 · 공단 기록과 99.6% 일치)과 다르지만, **앞으로의 구현은 백엔드를
따른다** — 등급이 두 곳에서 다르게 나오면 사용자가 먼저 다친다.

**단 `assessment` 는 그대로 둔다.** 이미 만들어졌고 호출부가 없어 충돌하지 않으며,
`stats/grade.py` 는 `coach` 의 `assess` 노드가 대상 요인을 고르는 데 계속 쓴다.
**coach 가 내보내는 등급 표기만 백엔드 값을 따른다.**

> 백엔드 임계값은 주석에도 `▲ 확정 필요` 이고 합의 명세도 확정 대기다. 우리 실측
> (`dev/AI-2` §5)을 넘겨 두면 확정할 때 근거가 된다.

---

## 5. 이 디렉터리의 문서가 지켜야 할 것

- **규약을 옮겨 적지 않는다.** `../0N` 의 절 번호를 가리킨다
- **산출물의 열 정의는 여기에 둔다.** `02` §4 가 "한 단계 = 한 CSV"라고만 정하고,
  그 CSV가 무엇인지는 기능마다 다르기 때문이다
- **한계와 산출되지 않은 칸을 적는다.** [AI-1](AI-1-age-band-distribution.md) §5 처럼,
  빠진 것을 사유와 함께 남긴다. 조용히 빠진 것은 나중에 결함으로 읽힌다
- **구현 전에 쓰고, 구현 뒤에 실측으로 고친다.** 계획 문서와 완료 문서를 나누지
  않는다 — 나누면 둘이 갈라진다
- **백엔드 스키마를 여기에 복사하지 않는다.** 정본은 `family-fitness-be` 의
  `docs/erd.dbml` 이고, 대조 결과만
  [AI-13](AI-13-backend-contract-reconciliation.md) 에 둔다
