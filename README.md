# 우리가족 체력키움 — AI 파트

가족의 체력 측정값을 국민체력100 또래 분포와 대조하고, 또래에게 실제로 처방된
운동을 찾아, 운동영상의 **클립**(유튜브를 끊은 것과 공단 국민체력100 동영상)으로 한 주
미션을 짜서 **제안**하는 서비스다.
승인과 저장은 하지 않는다 — 그건 백엔드 몫이다.

내는 것은 HTTP 다섯 자리뿐이다. 계약은 [`docs/인터페이스-명세.md`](docs/인터페이스-명세.md)
에 있고, 그 문서가 백엔드와의 약속이다.

---

## 기술 스택

| 자리 | 쓰는 것 | 왜 |
|---|---|---|
| 서비스 | **FastAPI** + uvicorn, **Pydantic** v2 | 요청 검증과 스키마를 한 곳에서 본다 |
| 검색 | **FAISS** (IndexFlatIP) + **bge-m3** 임베딩 | 청크 4,251개라 전수 비교가 싸다. 코사인 그대로 |
| 임베딩 | **llama-cpp-python** · 서비스 안에서 돈다 (`[embed]`) | 인덱스를 만든 바로 그 파일(`bge-m3-Q8_0.gguf`)을 읽어 벡터가 같게 나온다. 서버를 따로 띄우지 않는다 |
| LLM | **Claude** (`claude-opus-5`) · **Gemini** (`gemini-flash-latest`) | 한쪽이 붐비면 다른 쪽으로 넘어간다 |
| 표 만들기 | **pandas** · **numpy** · **openpyxl** | 원자료 67만 행 → 또래 분포·인증 기준 |
| 영상 쪼개기 | 미리 읽어 둔 **OCR 화면 글자** | 유튜브에 다시 가지 않는다. 새로 읽을 때만 `ocrmac`(`[collect]`) |
| 공단 동영상 | **공공데이터 API** (국민체력100 동영상) | 5분 미만 한 동작 영상을 끊지 않고 클립 하나로 쓴다. 체력요인·추천 체력수준도 API 값이다. 유튜브 클립과 한 목록에서 똑같이 고른다 |
| 품질 | **ruff** · **mypy** · **pytest** | `make verify` 하나로 돈다 |

파이썬 **3.11 이상**. 개발은 3.14 에서 했다.

---

## 빠르게 띄우기

```bash
# 1. 가상환경과 의존성 — [embed] 는 C++ 을 빌드해 몇 분 걸린다
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,embed]"

# 2. 키 채우기 — 이름과 뜻은 .env.example 에 있다
cp .env.example .env && $EDITOR .env

# 3. 임베딩 모델을 한 번 받아 두고 인덱스와 맞는지 잰다 (605 MB, 처음 한 번만)
make embed-model      # 「코사인 최소 1.00000」이 나오면 된다

# 4. 서비스 — 임베딩 서버를 따로 띄우지 않는다
make serve            # http://127.0.0.1:8000  · 문서는 /docs · 개발용(--reload)
```

`make serve` 는 코드를 고칠 때마다 다시 뜨므로 모델을 띄울 때 올리지 않고
(`EMBEDDING_WARMUP=0`) 검색이 처음 쓰일 때 올린다. 올리는 데 맥 Metal 에서는 1초
안쪽이었고, GPU 없는 Windows 에서는 2~10초 걸렸다. 운영에서는 띄울 때 올린다 —
아래 「운영에서 띄우기」.
밖에 띄운 llama.cpp 서버를 쓰고 싶으면 `.env` 에 `EMBEDDING_BACKEND=http` 를 적고
예전처럼 띄운다.

```bash
llama serve -hf gpustack/bge-m3-GGUF -hff bge-m3-Q8_0.gguf --embedding --port 8082
```

**맥에서 `[embed]` 빌드가 SDK 헤더 오류로 깨지면** 셸이 `CC`·`CXX` 를 Homebrew
LLVM 으로 잡아 둔 것이다. 애플 clang 으로 돌려 빌드한다.

```bash
CC=/usr/bin/clang CXX=/usr/bin/clang++ CMAKE_ARGS="-DGGML_METAL=on" pip install -e ".[dev,embed]"
```

띄운 뒤 한 바퀴 돌려 눈으로 보려면:

```bash
python scripts/probe.py
```

평가 → 또래 분포 → 영상 찾기 → 질문 → 한 주 편성까지 차례로 부르고, 어떤 클립이
몇 초짜리로 어디에 붙었는지 찍어 준다.

---

## git 에 없는 것

`data/release` 표 여섯 장과 `data/index`(`corpus.faiss` · `corpus.json` ·
`corpus_meta.csv`, 22MB)는 저장소에 들어 있다. 둘 다 서버가 띄울 때 있는지 보고,
하나라도 없으면 뜨지 않는다. 아래 둘은 빠져 있으니 따로 받는다.

| 무엇 | 자리 | 크기 | 받는 법 | 없으면 |
|---|---|---|---|---|
| 임베딩 모델 `bge-m3-Q8_0.gguf` | 허깅페이스 캐시(`gpustack/bge-m3-GGUF`). 다른 자리에 두려면 `.env` 의 `EMBEDDING_GGUF` | 605MB | `make embed-model` | 서버가 뜨지 않는다(`EMBEDDING_BACKEND=http` 가 아니면) |
| `data/raw/` (원자료 CSV · 기준표 · 화면 글자 · 프레임 · 공단 API 응답) | `data/raw/` | 2.2GB | 따로 받아 같은 자리에 둔다 | 표를 **다시 만들** 때만 필요하다 |

인덱스를 새로 만드는 코드는 이 저장소에 아직 없다. 지금 있는 `data/index` 는
코퍼스 4,251줄(처방 4,137 · 인증 기준 66 · 영상 48)로 만들어 둔 것이다.

---

## 운영에서 띄우기

`make serve` 는 개발용이다(`--reload`, 모델을 처음 쓸 때 올림). 운영에서는 이렇게 띄운다.

```bash
# 1. [dev] 없이 [embed] 만 — 시험 도구는 필요 없다
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[embed]"

# 2. 키 — ANTHROPIC_API_KEY · GEMINI_API_KEY (없으면 규칙 편성만 돈다)
cp .env.example .env && $EDITOR .env

# 3. 모델을 미리 받아 둔다(605MB). 띄울 때 받지 않는다 — 없으면 뜨지 않는다
make embed-model

# 4. --reload 없이 띄운다. AI_HOST 는 BE 만 닿는 주소로(아래)
make serve-prod AI_HOST=127.0.0.1 AI_PORT=8000
#    = python -m uvicorn family_fitness_ai.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

**워커는 하나로만 띄운다.** 편성 실행(`/v1/coach/runs`)은 그 프로세스 메모리에만
있다(`coach/runs.py` 의 `Store`). `--workers 2` 처럼 여럿으로 띄우면 BE 가 실행을 만든
워커와 결과를 묻는 워커가 달라져 `404 run_not_found` 가 나고, BE 는 AI 편성을 버리고
라벨 기반 대체 편성으로 빠진다. gunicorn 이나 컨테이너 설정으로 띄울 때도 워커 수
(`--workers`, `WEB_CONCURRENCY`)를 1 로 둔다. 같은 까닭으로 AI 를 여러 대 띄워 BE 앞에
나눠 붙이지도 않는다. 늘리려면 실행 저장소를 프로세스 밖으로 옮기는 설계 변경이 먼저다.

띄울 때 이것을 본다. 하나라도 어긋나면 서버가 뜨지 않고 까닭을 남긴다.

- `data/index` 세 파일과 `data/release` 표 여섯 장이 있는지
- 임베딩 모델이 있는지
- 모델을 올려 둔다(`EMBEDDING_WARMUP`, 기본 켜짐). 끄면 첫 검색 · 질문이 모델을
  올리느라 BE 의 읽기 한도(검색 4초)를 넘을 수 있다. 끄지 않는다.

**AI 포트는 밖에 열지 않는다. BE 만 부른다.** AI 서비스에는 인증이 없고,
`/v1/coach/runs` · `/v1/coach/messages` 는 LLM 을 부른다. 포트가 열려 있으면 누구나
AI 를 바로 불러 BE 가 걸어 둔 LLM 호출 한도(심사용 계정의 하루 편성 · 질문 횟수)를
건너뛰고 LLM 비용을 쓸 수 있다.

- BE 와 같은 기계면 `--host 127.0.0.1` 로 띄운다.
- 다른 기계면 사설망 주소로 띄우고, 방화벽 · 보안 그룹에서 BE 가 있는 곳만 그 포트에
  닿게 한다. `0.0.0.0` 으로 띄워 공개 포트에 붙이지 않는다.
- 띄운 뒤 바깥망에서 BE 의 `APP_AI_BASE_URL` 주소로 `/health` 를 불러 닿지 않는지 본다.

---

## 표 다시 만들기

서비스는 원자료를 읽지 않는다. `data/release` 표만 읽는다. 원자료가 바뀌었을
때만 아래를 차례대로 돌린다.

```bash
make tables        # 원자료 67만 행 → 또래 분포 1,746줄 · 인증 기준 1,122줄 · 등급 분포 1,048줄
make clips         # 화면 글자 → 클립 695개 (영상 48편, 길이 중앙값 42초)
make clip-labels   # 클립 이름 334개 → 처방 어휘·체력요인·단계·조건
make kspo          # 공단 동영상 → 5분 미만 한 편 = 클립 하나, 452개 (kspo_videos.csv)
```

`make kspo` 는 받아 둔 API 응답(`data/raw/kspo`)이 있으면 API 를 부르지 않는다.
처음이거나 다시 받으려면 `.env` 에 공공데이터포털 키(디코딩)를 `DATA_GO_KR_KEY` 로
적고 `make kspo KSPO_ARGS=--fetch`. 영상 파일이 열리는지도 그때 다시 본다 —
유아기 운동처방동영상 61편은 공단 서버에 파일이 없어(오류 페이지로 넘어간다) 빠진다.
API 한도는 개발계정 하루 10,000번이고, `--fetch` 한 번에 16번 부른다.

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

기준 경로 `/v1`, JSON(UTF-8), 인증 없다 — 그래서 BE 만 닿는 곳에 띄운다(「운영에서
띄우기」). 성공은 payload 를 그대로 내고 실패는
`{"error": {"code", "message"}}` 한 모양이다.

| 엔드포인트 | 하는 일 | 한도 |
|---|---|---|
| `POST /fitness/assessment` | 측정값을 또래와 대조해 요인별 점수로 | 3s |
| `POST /fitness/trajectory` | 또래 집단이 나이를 따라 보이는 분포 | 3s |
| `POST /videos/search` | 운동명·체력요인으로 영상 찾기 — 유튜브·공단을 한 번에 | 4s |
| `POST /coach/runs` → `GET /coach/runs/{run_id}` | 한 주 편성 (비동기) | 2s → 60s 안에 끝난다 |
| `POST /coach/messages` | 운동·체력 질문에 출처를 달아 답하기 | 10s |

자세한 요청·응답 모양과 오류 코드는 [`docs/인터페이스-명세.md`](docs/인터페이스-명세.md).

---

## 한 주 편성이 도는 차례

```
assess    측정값 → 또래 백분위 → 가장 낮은 요인        계산이다. LLM 이 끼지 않는다
          보호자가 키워 주고 싶은 역량(focus_factor)을 보냈으면 그 요인이 먼저다
          키울 요인은 「지금 키우기 좋은 영역」 · 「보호자가 키워 주고 싶은 역량」으로 부른다
retrieve  그 나이·성별·요인으로 처방된 운동 찾기        비면 한 칸씩 넓히고 notices 에 적는다
compose   후보 클립과 근거를 놓고 LLM 이 한 주를 짠다   목록 밖 id 는 버린다
          최근에 받은 영상(recent_video_ids)은 뒤로 미루고, 같은 순위끼리는 날짜로 섞는다
          최근 영상을 다시 쓸 때는 오래전에 받은 것부터, 한 회 안에서는 다른 영상부터
          focus_factor 가 있으면 본운동 칸의 4분의 3(올림)을 그 역량 클립으로 채운다
          요인별 등급(「심폐지구력 2등급」)은 글에 쓰지 않는다 — 쓰면 그 칸은 규칙 문구로
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
  common/    설정 · 오류 · release 표 목록 · 측정 항목 표 · 화면 문구 · LLM 호출
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
지킨다. `data/index` 나 임베딩(모델 또는 서버)이 없으면 그 시험만 건너뛴다 —
CI 는 `[embed]` 없이 돌아서 검색 시험을 건너뛴다.

---

## 알아 둘 것 셋

**LLM 이 없어도 돈다.** 키가 없거나, 붐비거나, 답이 규칙에 어긋나면 규칙 편성이
대신 선다. 그때는 `steps[2].status` 가 `partial` 로 나가 숨기지 않는다.

**AI 는 DB 에 쓰지 않는다.** 편성 결과는 메모리에만 있고 30분 뒤 사라진다.
저장과 승인은 전부 호출자가 한다.

**없으면 없다고 하지 않고 넓혀서 권한다.** 만 8세처럼 그 나이 처방 자료가 없으면
연령대·성별 순으로 넓히고 `notices` 에 적는다. 연령 라벨이 다른 영상이 섞여도
막지 않고 알린다 — 쓸지는 화면 저쪽에서 정한다. 어르신은 성인 영상을 또래로 친다
(어르신 전용 영상을 따로 두지 않는다).
