# AI-13 · 백엔드 스키마 대조와 계약 조정

`family-fitness-be` 의 ERD·ADR·유저플로를 읽고 `../03` 과 대조한 결과다.
**번호는 13이지만 순서는 첫 번째다** — 여기서 갈리는 것이 [AI-4](AI-4-assessment-endpoint.md)
이후 전부의 전제다.

| 읽은 것 | 위치 |
|---|---|
| ERD | `family-fitness-be` `docs/erd.dbml` (19 테이블 · 4 모듈) |
| 통합 결정 | `docs/adr/ADR-001-ddd-hexagonal-and-relational-data.md` |
| 화면 흐름·엔드포인트 | `docs/user-flow.md` |
| 구현 상태 | `backend/README.md` |

> **BE는 아직 부트스트랩만 있고 Flyway 마이그레이션 전이다** (`backend/README.md` —
> 네 모듈 전부 "Not started"). ERD는 초안이다. **지금이 가장 싸게 맞출 수 있는
> 시점이고, 그래서 이 문서가 급하다.**

`AGENTS.md` §2 대로 API 서버 코드는 건드리지 않는다. 필요한 변경은 여기 적어 알린다.

---

## 1. 해소된 미결 셋

**[README](README.md) §4 의 ①·②·③ 이 답을 얻었다.** 임의로 확정한 것이 아니라 BE
문서에 적혀 있는 값이다.

| 미결 | 답 | 근거 |
|---|---|---|
| ① 임베딩 모델과 차원 | **`vector(1536)` · `text-embedding-3-small`** | `ai_documents.embedding` · `embedding_model` |
| ② `missions` 실제 컬럼 | **§3.1 에 있다.** 우리 `proposal.missions[]` 와 다르다 | `missions` · `coach_run_proposal_items` |
| ③ AI가 pgvector에 직접 붙는가 | **붙는다. 단 `ai_documents` 하나로 제한된 DB 역할** | ADR-001 |

**②는 "답이 나왔다"가 아니라 "어긋난 것이 확인됐다"이다.** `../03` §11 ① 이
"①이 가장 급하다"고 한 이유가 그대로 현실이 됐다.

### 1.1 ①의 파장

`../01` §4.1 · `../04` §2 · `../04` §6 ① 이 전부 **1024를 전제로 쓰여 있다.**
1536으로 고치는 것은 숫자 하나가 아니다 — `../04` §5 대로 차원이 바뀌면 전량
재색인이고, 아직 색인한 것이 없으니 **지금 고치면 비용이 0이다.**

`text-embedding-3-small` 은 OpenAI 모델이다. `../01` §3.2 의 비밀값 표에는
`ANTHROPIC_API_KEY` 만 있고 OpenAI 키가 없다. **`.env.example` 과 `../01` §3.2 에
키 이름을 더해야 한다** (`AGENTS.md` §4 대로 이름만).

---

## 2. 우리가 고칠 것 — 전제가 틀렸다

**BE 스키마가 옳고 우리 문서가 낡은 것들이다.** 여기는 논의할 것이 없다.

### 2.1 공간·소음·기구 필터 — `400` 이 아니라 받아야 한다

`../03` §4.1 은 이렇게 못박았다.

> **공간·소음·기구·길이 필터는 받지 않는다.** 라벨이 그 필드를 갖지 않아 거를
> 데이터가 없다. 요청에 넣으면 무시가 아니라 `400` 이다

**그 전제가 사실이 아니다.** `exercise_videos` 는 `space`·`noise`·`intensity` 를
`not null` 로, `equipment` 를 nullable로 갖는다. 유저플로도 "RAG search filtering by
age, **noise level, space requirements**"라고 적는다.

**→ `../03` §4.1 을 고친다.** 필터를 받고, `../03` §12 의 "라벨 스키마가 그 필드를
라벨할 때" 조건이 충족된 것으로 본다.

### 2.2 AI 서비스의 DB 접근 범위

`../01` §4.2 는 `ai-batch` 가 `exercise_videos` 의 `ai_*` 컬럼에 쓴다고 했다.

**그런 컬럼은 없고, 그 권한도 없다.** `exercise_videos` 는 "Spring Boot 소유"이고
(ERD 주석), AI 역할은 `ai_documents` 하나로 제한된다 (ADR-001). 라벨링 결과는
DB에 직접 쓰는 것이 아니라 **BE에 넘겨야 한다.**

**→ `../01` §4.2 의 쓰기 경계표를 고친다.** `ai-service` 온라인 쓰기 없음은 그대로다.

### 2.3 스택 표기

| `../01` §2 | 실제 |
|---|---|
| Spring Boot 3 · Java 17 | **Kotlin · Spring Boot 4.1.1 · Java 25** |
| JWT (Spring Security) | **Google OAuth 단일 로그인** (`users.provider = GOOGLE`) |
| 컨테이너 3개 (`api-server`/`ai-service`/`ai-batch`) | BE는 **Spring Modulith 모듈러 모놀리스 1개** + 우리 서비스 |

우리 쪽 컨테이너 둘은 그대로 유효하다. **BE를 3컨테이너 그림에 그린 것만 틀렸다.**

### 2.4 FAISS 폴백의 지위

`../01` §1 은 "DB가 서기 전에도 개발이 멈추지 않게" FAISS를 두었다. ADR-001이
pgvector 직접 접속을 확정했으므로 **폴백이 아니라 로컬 개발 전용**이 된다.
`VECTOR_BACKEND` 는 남기되 배포 기본값은 `pgvector` 다.

---

## 3. BE에 알릴 것 — 스키마로는 계약을 지킬 수 없다

**여기가 이 문서의 본론이다.** 아래는 우리가 고쳐서 해결되지 않는다.

### 3.1 `missions` 에 우리 제안이 들어가지 않는다

`../03` §8 은 **"AI 응답을 가공하지 않는다. 승인 시 API 서버가 채우는 것은 소유권
컬럼뿐"** 이라고 정했다. 지금 스키마로는 불가능하다.

| 우리 `proposal.missions[]` (`../03` §5.6) | `missions` · `coach_run_proposal_items` | 결과 |
|---|---|---|
| `sessions[]` (요일·운동명·요인·분·영상·근거) | **없다.** 제안 1건 = 1행, `video_id` 단수 | 주 3회 세션이 저장되지 않는다 |
| `copy.child` · `copy.parent` | `description` 하나 | **아이/부모 문구 분리가 사라진다** |
| `participants[].role` (주행자·동반자·응원) | `mission_participants` 에 role 컬럼 없음 | 역할이 사라진다 |
| `evidence[]` · `citations[]` | `rationale varchar(400)` | **인용이 저장되지 않는다** |
| `fitness_factor` · `duration_min` | `target_metric`(STEPS/TIME) · `target_value` | 축이 다르다 |

**가장 심각한 것은 인용이다.** `../03` §8 은 `proposal.citations` 를 "missions 의
근거 컬럼"에 저장한다고 했고, `../01` §5 는 인용률을 **임계가 아니라 절대값 100%**
로 두었다. `rationale` 은 자유 텍스트라 인용이 아니다 — 어느 문서에서 나왔는지
대응이 없으면 검증할 수 없다.

`coach_message_citations` 는 대화용으로 이미 있다. **미션 쪽에 같은 모양이 필요하다**
(`coach_run_proposal_item_citations` 또는 그에 준하는 것).

> **제안** — `sessions`·`copy`·`participants` 를 컬럼으로 펴지 말고 JSONB 한 칸으로
> 받는 쪽이 싸다. `../03` §8 이 "컬럼으로 펼치면 제안과 저장본이 갈라진다"고 한
> 이유이고, 펴려면 표의 다섯 줄을 전부 컬럼으로 만들어야 한다.

### 3.2 `coach_runs` 에 `steps` 와 거부가 없다

| 우리 (`../03` §5.3) | `coach_runs` | 결과 |
|---|---|---|
| `steps[]` (4단계 감사 기록) | `summary text` | **어느 단계가 강등됐는지 안 남는다** |
| `status: refused` | `RUNNING`·`APPROVED`·`REJECTED` | **거부가 저장되지 않는다** |
| `run_id` = `cr_` 접두 문자열 | `id uuid` | 우리가 uuid로 맞추면 된다 |

**`REJECTED` 는 보호자가 거절한 것이고, 우리 `refused` 는 AI가 근거를 못 찾은
것이다. 전혀 다르다.** 둘을 한 값에 넣으면 `../01` §5 의 거부율 지표가 "보호자가
싫어한 비율"과 섞여 측정 불가가 된다.

같은 이유로 `coach_messages` 에도 `refused`·`refusal_reason` 이 필요하다.
`../03` §8: **"거부 응답도 저장한다 — 거부율이 품질 지표다."**

### 3.3 `ai_documents` 로는 연령 안전 규칙을 걸 수 없다

**이것이 가장 급한 하나다.** `../04` §3 · `../03` §4.1 이 반복해 적은 것 —
연령 필터는 정확성이 아니라 **안전** 문제다.

`ai_documents` 에는 **`age_from` 하나뿐이고 `age_to` 가 없다.**

상한이 없으면 "성인 전용"을 표현할 방법이 없다. `age_from = 19` 인 성인 처방이
연령 하한 조건만으로는 걸러지지 않거나, 걸러려면 애플리케이션 코드가 판단해야 한다.
**`../04` §3 은 필터가 질의의 `WHERE` 절에 있고 거치지 않는 경로가 없어야 한다고
정했다.** 컬럼이 없으면 그 보장이 코드 관례로 내려앉는다.

| 필요한 것 | 왜 |
|---|---|
| `age_to smallint` | 상한 없이는 성인 자료를 아이에게서 막지 못한다 |
| `age_group varchar` 또는 그에 준하는 것 | 우리 연령 구분은 다섯 개 한국어 값이고 나이 범위와 1:1이 아니다 (`../02` §2.1) |
| `null` 의 뜻 | `../04` §2.2 는 `age_group IS NULL` 을 "연령 무관"으로 쓴다. `age_from` 이 nullable인데 그 뜻이 정의돼 있지 않다 |

### 3.4 `ai_documents` 에 `chunk_id` 가 없다

`../04` 전체가 여기 걸려 있다.

> `citation.label` 과 `chunk_id` 는 API를 넘어가 `missions.evidence` 에 저장되므로,
> **만드는 규칙이 없으면 같은 자료가 재색인 때마다 다른 인용이 된다.**

`ai_documents` 의 PK는 `gen_random_uuid()` 다. **재색인하면 새 uuid가 나온다** —
`content_hash` 유니크 제약이 중복 삽입은 막지만, upsert가 아니라 새 행이 되면 저장된
`ai_document_id` 인용이 끊긴다.

| 우리 (`../04` §1) | `ai_documents` |
|---|---|
| `chunk_id text PK` = `{source}:{natural_key}` · 결정적 | `id uuid` 랜덤 + `content_hash char(64)` |
| `corpus_version` | `embedding_version` (뜻이 다르다) |
| `metadata jsonb` | 컬럼으로 폄 (ADR-001의 의도적 선택 — 여기는 동의한다) |

**두 안 중 하나면 된다.**

1. `ai_documents` 에 `chunk_id text unique` 를 더하고 우리가 채운다. 인용 필드는
   그대로 `ai_document_id` 를 쓰되 우리 재색인이 같은 행을 upsert한다
2. `content_hash` 를 우리 `natural_key` 해시로 정의하고 **upsert 규칙을 문서로 못박는다**

1을 권한다 — `../04` §1 이 요구하는 "인용이 이어진다"를 스키마가 보장한다.

### 3.5 측정 항목의 키가 이름이다

`fitness_test_items.item varchar(30)` 이 `악력`·`유연성` 같은 **이름**이다.
`../03` §3.1 이 정확히 이 문제를 경고했다.

> 이름을 키로 쓰지 않는 이유는 표기가 흔들리기 때문이다 — 기준표는 `제자리멀리뛰기`,
> 컬럼정의서는 `제자리 멀리뛰기` 다. **코드가 식별자이고 이름은 표기다.**

BE 유저플로도 "Fitness item code mappings ... remain pending validation"으로 이것을
미결로 두고 있다. **`../02` §5.1 의 요인×연령대 기준항목 표와 `../03` §9 의 코드표가
그 답이다.** `item` 을 3자리 코드로 바꾸거나 `item_code` 를 따로 두기를 제안한다.

같은 표에 **`score` 컬럼이 없다.** `percentile`·`grade` 는 있다. `../02` §5 전체가
요인 점수 0~100을 만드는 규약이고 `../03` §3.5 의 `factors[].score` 가 그 값인데,
저장할 자리가 없으면 화면이 매번 우리를 다시 부르거나 점수를 버려야 한다.

### 3.6 `exercise_videos` 의 `not null` 이 더미를 강요한다

`age_from`·`age_to`·`intensity`·`space`·`noise` 가 전부 `not null` 이다.

**공단 채널 전수 362편 중 314편(87%)이 연령 추정 불가다** (`../02` §2.3 · 실측).
`not null` 이면 그 314편은 **넣을 수 없거나, 지어낸 값을 넣어야 한다.**

`AGENTS.md` §4 — "데이터가 없을 때 더미로 채워 '돌아가게' 만들지 않는다".
그리고 지어낸 연령은 `../02` §2.3 이 막으려던 바로 그 사고다 — 아이에게 성인 영상.

**→ `age_from`·`age_to`·`space`·`noise` 를 nullable로. `null` 은 "라벨 없음"이고,
라벨 없는 영상은 아이 프로필에 나가지 않는다** (`../02` §2.3).

### 3.7 `predictions` 는 우리가 하지 않기로 한 것이다

`../03` §7 은 명시적으로 금지했다.

> **응답 필드에 `predict`·`forecast` 계열 이름을 두지 않는다.** 횡단면 자료라
> 개인의 미래를 다루지 않는다.

BE는 `predictions` · `prediction_points` · `years_from_now` · `scenario`
(MAINTAIN/IMPROVE)다. `p10`/`p50`/`p90` 구조는 우리 `bands[]` 와 그대로 맞으므로
**모양은 맞고 이름과 주장이 다르다.**

`scenario` 는 우리에게 없다. 우리는 집단 분포 하나를 낸다 — "개선 시나리오"를 내려면
개인의 변화율을 가정해야 하고, 횡단면 자료에는 그 근거가 없다.

**→ 이름을 `cohort_bands` 계열로 바꾸고 `scenario` 를 빼거나, `MAINTAIN` 만 두기를
제안한다.** `../03` §7.2 의 `notice` 를 화면에 그대로 노출하는 것은 그대로다.

### 3.8 run의 단위와 트리거

| | 우리 (`../03` §5) | BE |
|---|---|---|
| 단위 | 프로필 (`409` 도 프로필 기준) | **가족·주차** (`uq_coach_run (family_id, week_start)`) |
| 트리거 | 사용자가 "이번 주 미션 받기" | **일요일 20시 CRON** + MANUAL |

**BE 쪽이 맞다.** 편성 입력이 `profile_refs[]` 1~4명이니 실제 단위는 가족이다.
`../03` §5 의 409 문구를 가족·주차 기준으로 고쳐야 한다.

CRON은 우리에게 새 요구다 — 예상 15~40초짜리 호출이 일요일 20시에 가족 수만큼
몰린다. `../01` §2.3 의 오토스케일과 `ai-service` 태스크 2개로 감당되는지는
[AI-12](AI-12-deploy.md) 에서 다시 본다.

---

## 4. 엔드포인트 이름

유저플로의 BE 엔드포인트는 클라이언트용(`/api/v1/...`)이고 우리 것(`/v1/...`)과
층이 다르다. **겹치는 것은 하나뿐이다.**

| 유저플로 | 우리 (`../03` §6) |
|---|---|
| `POST /api/v1/coach/chat` | `POST /v1/coach/messages` |

BE가 자기 클라이언트 경로를 무엇으로 하든 우리 경로와 무관하다. **다만 같은 것을
다르게 부르면 대화가 어긋나므로 한쪽으로 맞추기를 제안한다.**

`POST /api/v1/coach/runs/{id}/approve|reject` 가 승인 게이트다 — `../03` §8 의
③④ 가 여기다. **AI에 미션을 만들 경로가 없다는 우리 보장은 그대로 유효하다.**

---

## 5. 맞은 것

어긋난 것만 적으면 무엇을 안 고쳐도 되는지 알 수 없다.

| 항목 | 양쪽 |
|---|---|
| 승인 게이트 | AI는 제안만, 저장·승인은 BE. ADR-001과 `../03` §1 이 같다 |
| 인용 필수 | "responses must reference `ai_document_id` or are considered bugs" = `../01` §5 인용률 100% |
| 강점 기준 75백분위 | 유저플로 = 우리 `Band.strength` (`../03` §2.4) |
| 영상 메타는 BE가 조인 | `exercise_videos` 가 title·duration 보유 = `../03` §1 |
| 내용 해시로 중복 방지 | `content_hash` SHA-256 = `../04` §6 ④ 의 우리 기본값 |
| 프로필은 불투명 참조 | `profiles.id uuid` = `../03` §2.4 `ProfileRef` — **§11 ② 해소** |
| p10/p50/p90 밴드 | `prediction_points` = `../03` §7.2 `bands[]` (이름만 §3.7) |

---

## 6. 새로 생긴 것

BE에는 있고 우리 계약에 없다. **지금 정할 필요는 없으나 모르고 있으면 안 된다.**

| 항목 | 무엇 | 우리에게 |
|---|---|---|
| `activity_daily` | 걸음수·활동시간 | `missions.target_metric = STEPS` 의 근거. 우리 미션 모델과 축이 다르다 (§3.1) |
| `video_interactions` | 시청·좋아요·진도율 | `../03` §12 "최근 수행 이력 입력"의 재료가 이미 있다 |
| `conversation_id` | 대화 세션 | 우리는 단발 질의응답이다 (`../03` §6). 문맥을 받을지 정해야 한다 |
| `cheers` | 응원 스티커·메시지 | 우리 `Role.응원` 과 이름만 같고 다른 것이다 |
| 만 4세 미만 프로필 | 측정 항목이 없다 | `../02` §2.2 의 만 7~10세 공백과 같은 처리 — `factors: []` |
| 미션 인증 | 영상 90% 재생 · 타이머 · 자가보고 | 우리 `sessions[].duration_min` 과 이어질 자리 |

---

## 7. 무엇부터

1. **§3.3 연령 상한** — 안전 문제라 먼저다. 스키마 한 컬럼이다
2. **§3.4 `chunk_id`** — 색인을 시작하기 전에 정해야 한다. 시작한 뒤면 재색인이다
3. **§3.1 미션 인용 컬럼** — [AI-11](AI-11-coach-runs.md) 이 여기 걸려 있다
4. **§3.2 거부 저장** — 없으면 `../01` §5 의 품질 지표를 못 낸다
5. §3.5 항목 코드 · §3.6 nullable · §3.7 이름 — 순서는 BE 편의대로

§1 의 셋(임베딩 1536 · pgvector 직접 접속 · profile_ref uuid)은 **오늘 우리 문서에
반영할 수 있다.** 나머지는 합의가 먼저다.

---

## 8. 미결

| # | 항목 | 상태 | 확정 전 기본값 |
|---|---|---|---|
| ① | `../03` 을 어디까지 고칠 것인가 | **합의 대기** | 고치지 않는다. `AGENTS.md` §3 — 계약 필드 변경은 멈추고 묻는다 |
| ② | 미션 제안을 JSONB로 받을지 컬럼으로 펼지 | BE 결정 | JSONB (§3.1 제안) |
| ③ | `ai_documents` 를 우리가 upsert하는가 | ADR과 `backend/README` 표기가 다르다 | 우리가 유일한 쓰기 주체 (`backend/README`) |
| ④ | OpenAI 키 이름 | 미정 | `OPENAI_API_KEY` — `.env.example` 에 이름만 (§1.1) |
| ⑤ | CRON 동시 실행 규모 | 미정 | 가족 수 미상. [AI-12](AI-12-deploy.md) 에서 다시 본다 (§3.8) |
