# dev · 구현 순서

`01`~`04` 가 **무엇을 만드는가**를 정한다. 이 디렉터리는 **그것을 어떤 순서로,
무엇을 산출하며 만드는가**를 적는다. 한 파일이 한 이슈다.

규약을 여기에 복사하지 않는다. 규약이 필요하면 `../0N` 을 가리킨다.

---

## 1. 지금 어디까지 왔나

| 구역 | 상태 |
|---|---|
| `docs/01`~`04` | 확정. **일부가 백엔드 스키마와 어긋난다** — [AI-13](AI-13-backend-contract-reconciliation.md) |
| `stats/` · `ingest/` | **동작한다.** 원자료 → 점수 눈금 → 산출물 4종 → 채점 CLI |
| `api/` · `graph/` · `rag/` · `labeling/` · `common/` | `__init__.py` 만 있다 |
| 의존성 | `numpy` · `pandas` · `openpyxl` 셋뿐. FastAPI도 LangGraph도 아직 없다 |
| CI | `ruff` · `mypy` · `pytest` 가 `main`·`develop`·PR 에서 돈다 |

계약(`03`)에 정의된 엔드포인트 넷 중 구현된 것은 0개다.

**백엔드(`family-fitness-be`)도 부트스트랩만 있다** — 네 모듈 전부 "Not started",
Flyway 마이그레이션 전, ERD는 초안이다. 양쪽 다 아직 안 굳었다는 뜻이고, **그래서
지금이 계약을 맞출 가장 싼 시점이다.**

---

## 2. 순서와 의존

```
                    AI-13 백엔드 계약 대조 ★ 먼저
                     │
AI-1 분포 산출 ✅    │
 │                   │
 ├─ AI-2 등급 판정·또래 분포 ─┐
 │                            ├─ AI-4 assessment ─────────┐
 ├─ AI-5 궤적 밴드 ───────────┤                            │
 │                            │                            │
AI-3 서비스 골격 ─────────────┴─ AI-9 videos/search ─┐     │
                                                     │     │
AI-6 처방 코퍼스 ─┐                                  │     │
                  ├─ AI-8 임베딩·색인·검색 ──────────┼─────┤
AI-7 영상 라벨링 ─┘                                  │     │
                                                     └─ AI-10 coach/messages
                                                           │
                                                     AI-11 coach/runs
                                                           │
                                                     AI-12 배포·관측
```

| 이슈 | 무엇 | 선행 | 막고 있는 것 |
|---|---|---|---|
| [AI-13](AI-13-backend-contract-reconciliation.md) ★ | 백엔드 스키마 대조와 계약 조정 | — | **합의 대기 (§4)** |
| [AI-1](AI-1-age-band-distribution.md) ✅ | 연령 구간별 점수 분포 | — | — |
| [AI-2](AI-2-grade-card.md) | 등급 판정과 또래 등급 분포 | AI-1 | 신체조성 문턱 (§4 ④) |
| [AI-3](AI-3-service-skeleton.md) | FastAPI 골격·설정·오류·로그 | — | — |
| [AI-4](AI-4-assessment-endpoint.md) | `POST /fitness/assessment` | AI-2·3 | 저표본 처리 (`02` §6 ③) |
| [AI-5](AI-5-trajectory-bands.md) | `POST /fitness/trajectory` | AI-3 | 밴드 축 (`02` §6 ②) |
| [AI-6](AI-6-prescription-corpus.md) | 처방 어휘·청크 추출 | AI-1 | 일련 규칙 (`04` §6 ④) |
| [AI-7](AI-7-video-labeling.md) | 영상 수집과 라벨링 | — | 영상 라벨 스키마 (§4 ②) |
| [AI-8](AI-8-vector-index.md) | 임베딩·색인·검색·임계값 | AI-6·7 | **`ai_documents` 스키마 (§4 ①②)** |
| [AI-9](AI-9-videos-search.md) | `POST /videos/search` | AI-3·8 | — |
| [AI-10](AI-10-coach-messages.md) | `POST /coach/messages` | AI-8·9 | 거부 저장 (§4 ③) |
| [AI-11](AI-11-coach-runs.md) | 미션 편성 그래프 | AI-4·10 | **미션 인용 컬럼 (§4 ③)** |
| [AI-12](AI-12-deploy.md) | 컨테이너·배포·관측 | AI-3 | — |

---

## 3. 묶음

**한 묶음이 끝나면 그것만으로 보여줄 것이 있다.** 마지막에 한 번에 합치지 않는다.

| 묶음 | 이슈 | 끝나면 보여줄 수 있는 것 |
|---|---|---|
| **M0 · 계약을 맞춘다** | AI-13 | 양쪽 스키마가 같은 것을 뜻한다 |
| **M1 · 측정값이 화면에 닿는다** | AI-2·3·4·5 | 나이·성별·측정값을 넣으면 점수·등급·또래 분포·궤적이 계약 모양으로 나온다 |
| **M2 · 근거가 생긴다** | AI-6·7·8 | 처방·영상·기준 청크가 색인되고, 질의에 근거가 붙어 나온다 |
| **M3 · 코치가 답한다** | AI-9·10·11 | 영상 검색·질의응답·주간 미션 제안이 인용과 함께 돈다 |
| **M4 · 배포** | AI-12 | 두 컨테이너가 ECS에서 돌고 `/readyz` 가 준비성을 판단한다 |

**M1은 백엔드 합의를 기다리지 않는다.** 계산과 산출물이 전부 우리 안에 있고,
바뀔 수 있는 것은 응답을 감싸는 껍데기뿐이다. **M2부터가 `ai_documents` 스키마에
걸리므로, 합의를 기다리는 동안 M1을 끝내 둔다.**

---

## 4. 지금 답이 필요한 것

미결의 정본은 각 문서다. 여기서는 **무엇이 무엇을 막고 있는지만** 모은다.
`AGENTS.md` §3 대로 임의로 확정하지 않는다.

| # | 답이 필요한 것 | 정본 | 막는 것 | 확정 전 기본값 |
|---|---|---|---|---|
| ① | **`ai_documents` 에 연령 상한이 없다** | [AI-13](AI-13-backend-contract-reconciliation.md) §3.3 | AI-8 · **안전 규칙** | 없음 — 상한 없이는 `04` §3 을 못 지킨다 |
| ② | **`ai_documents` 에 `chunk_id` 가 없다** | [AI-13](AI-13-backend-contract-reconciliation.md) §3.4 | AI-8 색인 시작 | `chunk_id text unique` 추가 요청 |
| ③ | **미션 인용 컬럼 · 거부 저장** | [AI-13](AI-13-backend-contract-reconciliation.md) §3.1·§3.2 | AI-10 · AI-11 | 없음 — 인용률·거부율을 못 낸다 |
| ④ | 신체조성 등급 문턱 | [AI-2](AI-2-grade-card.md) §5 | AI-2 등급 판정 | 3등급을 건강체력만으로 판정 |
| ⑤ | 저표본(`n < 30`) 칸의 응답 | `02` §6 ③ | AI-4 응답 | `score`·`percentile` 을 `null` |
| ⑥ | 궤적 밴드의 축 | `02` §6 ② | AI-5 산출물 | raw 밴드 |
| ⑦ | `SIM_THRESHOLD` 값 | `04` §6 ② | AI-8 · `/readyz` | 없음 — 값이 없으면 기동하지 않는다 |

### 4.1 답이 나온 것

**[AI-13](AI-13-backend-contract-reconciliation.md) §1 에서 셋이 해소됐다.**
백엔드 문서에 적혀 있는 값이지 우리가 정한 것이 아니다.

| 옛 미결 | 답 |
|---|---|
| 임베딩 모델과 차원 (`01` §7 ② · `04` §6 ①) | **`vector(1536)` · `text-embedding-3-small`** |
| `missions` 실제 컬럼 (`03` §11 ①) | **[AI-13](AI-13-backend-contract-reconciliation.md) §3.1 — 우리 제안이 들어가지 않는다** |
| AI가 pgvector에 직접 붙는가 (`01` §7 ① · `03` §11 ③) | **붙는다. `ai_documents` 하나로 제한된 역할** |
| `profile_ref` 형식 (`03` §11 ②) | **`profiles.id` uuid** |

`01`~`04` 본문은 아직 1024와 옛 전제로 쓰여 있다. **고치는 것은
[AI-13](AI-13-backend-contract-reconciliation.md) §8 ① 의 합의 뒤다** — 계약
필드를 바꾸는 일이라 `AGENTS.md` §3 대로 멈춘다.

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
