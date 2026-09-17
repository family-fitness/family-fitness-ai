# 우리가족 체력키움 — AI 파트

가족의 체력 측정값을 국민체력100 또래 분포와 대조하고, 또래에게 실제로 처방된
운동을 찾아, 공단 운동영상의 **클립**으로 한 주 미션을 짜서 **제안**하는 서비스다.
승인과 저장은 하지 않는다 — 그건 백엔드 몫이다.

내는 것은 HTTP 다섯 자리뿐이다. 계약은 [`docs/인터페이스-명세.md`](docs/인터페이스-명세.md)
에 있고, 그 문서가 백엔드와의 약속이다.

---

## 기술 스택

| 자리 | 쓰는 것 | 왜 |
|---|---|---|
| 서비스 | **FastAPI** + uvicorn, **Pydantic** v2 | 요청 검증과 스키마를 한 곳에서 본다 |
| 검색 | **FAISS** (IndexFlatIP) + **bge-m3** 임베딩 | 청크 4,251개라 전수 비교가 싸다. 코사인 그대로 |
| 임베딩 서버 | **llama.cpp** (`--embedding`) · 밖에서 띄운다 | 인덱스를 만들 때 쓴 모델과 같아야 한다 |
| LLM | **Claude** (`claude-opus-5`) · **Gemini** (`gemini-flash-latest`) | 한쪽이 붐비면 다른 쪽으로 넘어간다 |
| 표 만들기 | **pandas** · **numpy** · **openpyxl** | 원자료 67만 행 → 또래 분포·인증 기준 |
| 영상 쪼개기 | 미리 읽어 둔 **OCR 화면 글자** | 유튜브에 다시 가지 않는다. 새로 읽을 때만 `ocrmac`(`[collect]`) |
| 품질 | **ruff** · **mypy** · **pytest** | `make verify` 하나로 돈다 |

파이썬 **3.11 이상**. 개발은 3.14 에서 했다.

---

## 빠르게 띄우기

```bash
# 1. 가상환경과 의존성
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. 키 채우기 — 이름과 뜻은 .env.example 에 있다
cp .env.example .env && $EDITOR .env

# 3. 임베딩 서버 (검색·질문·편성에 필요하다)
llama serve -hf gpustack/bge-m3-GGUF -hff bge-m3-Q8_0.gguf --embedding --port 8082

# 4. 서비스
make serve            # http://127.0.0.1:8000  · 문서는 /docs
```

띄운 뒤 한 바퀴 돌려 눈으로 보려면:

```bash
python scripts/probe.py
```

평가 → 또래 분포 → 영상 찾기 → 질문 → 한 주 편성까지 차례로 부르고, 어떤 클립이
몇 초짜리로 어디에 붙었는지 찍어 준다.

---

## git 에 없는 것

`data/release` 표는 저장소에 들어 있다. 아래 둘은 용량 때문에 빠져 있으니 따로
받아서 같은 자리에 둔다.

| 자리 | 크기 | 없으면 |
|---|---|---|
| `data/index/` (`corpus.faiss` · `corpus.json` · `corpus_meta.csv`) | 22MB | 검색·질문·편성이 안 돈다 |
| `data/raw/` (원자료 CSV · 기준표 · 화면 글자 · 프레임) | 2.2GB | 표를 **다시 만들** 때만 필요하다 |

인덱스를 새로 만드는 코드는 이 저장소에 아직 없다. 지금 있는 `data/index` 는
코퍼스 4,251줄(처방 4,137 · 인증 기준 66 · 영상 48)로 만들어 둔 것이다.

---

## 표 다시 만들기

서비스는 원자료를 읽지 않는다. `data/release` 표만 읽는다. 원자료가 바뀌었을
때만 아래를 차례대로 돌린다.

```bash
make tables        # 원자료 67만 행 → 또래 분포 1,746줄 · 인증 기준 1,122줄 · 등급 분포 1,048줄
make clips         # 화면 글자 → 클립 531개 (영상 48편, 길이 중앙값 52초)
make clip-labels   # 클립 이름 264개 → 처방 어휘·체력요인·단계·조건
```

`make clip-labels` 만 LLM 을 부른다. 이미 붙여 둔 이름은 다시 묻지 않고, 사람이
고친 줄(`source=human`)은 `--refresh` 를 줘도 지킨다. 이름을 잇는 방법은 셋이고
그 경계는 재 보고 정했다 — 글자가 같으면 `exact`, 임베딩이 0.90 이상이면 `embed`,
그 아래는 임베딩이 「캐치볼을 해요」에 「낚시를 해요」를 붙이는 식으로 자주 틀려서
후보만 추려 LLM 에게 고르게 한다.

의료 질의를 얼마나 맞게 가리는지도 잴 수 있다:

```bash
make medical       # 고정 질의 20건으로 오탐·누락을 센다
```

---

## 내는 것

기준 경로 `/v1`, JSON(UTF-8), 인증 없다. 성공은 payload 를 그대로 내고 실패는
`{"error": {"code", "message"}}` 한 모양이다.

| 엔드포인트 | 하는 일 | 한도 |
|---|---|---|
| `POST /fitness/assessment` | 측정값을 또래와 대조해 요인별 점수로 | 3s |
| `POST /fitness/trajectory` | 또래 집단이 나이를 따라 보이는 분포 | 3s |
| `POST /videos/search` | 운동명·체력요인으로 영상 찾기 | 4s |
| `POST /coach/runs` → `GET /coach/runs/{run_id}` | 한 주 편성 (비동기) | 2s → 60s 안에 끝난다 |
| `POST /coach/messages` | 운동·체력 질문에 출처를 달아 답하기 | 10s |

자세한 요청·응답 모양과 오류 코드는 [`docs/인터페이스-명세.md`](docs/인터페이스-명세.md).

---

## 한 주 편성이 도는 차례

```
assess    측정값 → 또래 백분위 → 가장 낮은 요인        계산이다. LLM 이 끼지 않는다
retrieve  그 나이·성별·요인으로 처방된 운동 찾기        비면 한 칸씩 넓히고 notices 에 적는다
compose   후보 클립과 근거를 놓고 LLM 이 한 주를 짠다   목록 밖 id 는 버린다
verify    인용·금지 어휘를 보고 내보낸다                부분 통과는 없다
```

**영상은 통째로 내보내지 않는다.** 한 편에 운동이 여럿 들어 있어서, 화면에 뜨는
운동 이름이 바뀌는 지점마다 끊어 둔 클립(대개 1분 안쪽)을 골라
`준비운동 → 본운동 → 정리운동` 순으로 잇는다. 15분 한 회면 클립 12~13개가 붙는다.

---

## 디렉터리

```
src/family_fitness_ai/
  api/       FastAPI 앱과 요청 스키마
  coach/     편성(compose) · 질문 답(answer) · 검증(verify) · 실행 보관(runs)
  common/    설정 · 오류 · 측정 항목 표 · 화면 문구 · LLM 호출
  rag/       임베딩 · 인덱스 · 검색 · 의료 질의 가리기
  stats/     원자료 → 표(build) · 표 읽기(tables) · 평가(assess)
  video/     클립 끊기(clips) · 이름 붙이기(labels) · 목록과 편성(catalog)
docs/        인터페이스 명세
scripts/     probe.py — 돌고 있는 서비스를 눈으로 확인
tests/       시험. LLM 을 부르지 않는다
```

---

## 확인

```bash
make verify                            # 가상환경을 켜 두었을 때
make verify PY=.venv/bin/python        # 안 켰을 때
```

`lint`(ruff) · `types`(mypy) · `test`(pytest) 셋을 돌린다. CI 가 도는 것과 같다.

test은 **LLM 을 부르지 않는다.** 부르면 돈이 들고, 답이 매번 달라 test 노릇을
못 한다. LLM 이 없을 때의 길(규칙 편성)이 늘 서 있어야 한다는 것도 여기서 같이
지킨다. `data/index` 나 임베딩 서버가 없으면 그 시험만 건너뛴다.

---

## 알아 둘 것 셋

**LLM 이 없어도 돈다.** 키가 없거나, 붐비거나, 답이 규칙에 어긋나면 규칙 편성이
대신 선다. 그때는 `steps[2].status` 가 `partial` 로 나가 숨기지 않는다.

**AI 는 DB 에 쓰지 않는다.** 편성 결과는 메모리에만 있고 30분 뒤 사라진다.
저장과 승인은 전부 호출자가 한다.

**없으면 없다고 하지 않고 넓혀서 권한다.** 만 8세처럼 그 나이 처방 자료가 없으면
연령대·성별 순으로 넓히고 `notices` 에 적는다. 연령 라벨이 다른 영상이 섞여도
막지 않고 알린다 — 쓸지는 화면 저쪽에서 정한다.
