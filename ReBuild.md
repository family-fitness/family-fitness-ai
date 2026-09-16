# ReBuild — 문서와 CSV 를 미션 중심으로 재편한다

**무엇을 만드는가**는 [`docs/05`](docs/05) 에 있다.
이 문서는 **그것을 담을 그릇을 어떻게 다시 짜는가**만 적는다 — 문서 구조와 CSV 구조.

지금 산출물은 **파이프라인 단계별**로 나뉘어 있다 (분포 → 눈금 → 어휘 → 청크 → 색인).
미션 중심이면 **미션이라는 물건을 축으로** 묶여야 한다. 그 이사 작업의 명세다.

훑은 결과 — 문서 절·CSV·코드 **163개 항목** 중 **49개를 바꿔야 하고 4개는 아무도 쓰지
않는다.** 아래 작업은 그 목록에서 나왔다.

---

## 0. 먼저 알아야 할 것

**미션은 청크가 아니라 표의 한 행이다.** 그래서 `docs/04`(RAG 코퍼스 규약)가 통째로
미션에 적용되지 않는다. `coach/messages`(질문 답변)는 벡터 검색을 그대로 쓰므로
**`docs/04` 는 살아남되 범위가 좁아진다.**

**와이어에 자리가 없는 값을 우회해 끼워 넣지 않는다.** 자리가 아예 없는 것은 둘이다 —
`end_sec` 와 구간별 라벨 ([`AI-14`](docs/05) §3.2 ㉢㉤ ·
[`AI-11`](docs/05) §2.1 의 슬롯 대조표). 나머지는 계약대로 내고 진행한다.
`docs/03` 은 팀 합의 명세라 우리가 임의로 고치지 않는다 — `AI-13` §9 에 올려 반영한다.

---

## 1. CSV 재편

### 1.1 지금 있는 것 — 14개

| 파일 | 행 | 누가 읽나 | 미션에서 |
|---|---|---|---|
| `release/exercise_vocabulary.csv` | 641 | `labeling/label.py` | **바꾼다** → §1.2 ① |
| `release/grade_thresholds.csv` | 1,036 | `stats/assess` · `rag/chunks` | 그대로 |
| `release/age_band_score_summary.csv` | 394 | assessment | 그대로 |
| `release/age_band_score_distribution.csv` | 3,890 | assessment | 그대로 |
| `release/age_band_grade_distribution.csv` | 176 | assessment | 그대로 |
| `release/age_band_value_quantiles.csv` | 389 | assessment | 그대로 |
| `release/body_composition_ranges.csv` | 78 | assessment | 그대로 |
| `release/sim_queries.csv` | 48 | `rag/threshold` | `coach/messages` 용으로 남는다 |
| `interim/prescription_chunks.csv` | 708 | `rag/chunks` | **바꾼다** → §1.2 ② |
| `interim/video_labeling.csv` | 43 | `rag/chunks` | **바꾼다** → §1.2 ④ (이름은 다른 갈래가 코드에서 바꿨다) |
| `interim/video_exercises.csv` | 142 | `rag/chunks` | **바꾼다** → §1.2 ③ |
| `interim/videos.csv` | **43** | `labeling/label` | 그대로 (수집 메타) |
| `release/chunks.csv` | **822** | `rag/index` | **`interim` 에서 옮겼다** — 새 클론에서 색인을 다시 굽는 재료다 |
| `interim/sim_eval.csv` | 2,408 | 리포트 | `coach/messages` 용으로 남는다 |

> **`videos.csv` 는 43행이다.** 앞서 적은 304 는 `wc -l` 값(305)에서 1을 뺀 것이고,
> `description` 에 줄바꿈이 있어 행 수가 아니다 (`pandas.read_csv` 로 세면 43).
> **채널 전체를 수집한 적이 없다** — `data/raw/youtube/api.jsonl` 에 `channels.list`
> 호출이 0건이고, **두 재생목록만 수집했다.** 「수집 304 대 라벨 43」이라는 차이는 없다.
> 채널 전수와 라벨 가능한 상한은 `docs/02` §2.3 에 있다 (합집합 519편 · 라벨 상한 61편).

### 1.2 새로 낼 것 — 5개

**`data/release/` 에 둔다.** 미션 표는 백엔드가 적재할 산출물이라 커밋한다
(`.gitignore` 가 `data/*` 를 무시하고 `data/release/` 만 예외다).

#### ① `exercises.csv` — 운동

**이 운동이 공공데이터의 어느 값들을 나타내는지**를 한 줄에서 본다 (AI-14 §4.1).

| 열 | 무엇 |
|---|---|
| `exercise_id` | 결정적 id. 이름에서 만든다 (`identity()`) |
| `exercise_name` | 대표 이름 |
| `raw_forms` | **덮는 원자료 문자열 전부.** 세미콜론 |
| `merge_reason` | `표기`\|`동의어`\|`부위통합` — 무엇으로 합쳤는지 |
| `phase` | `준비운동`\|`본운동`\|`정리운동` — **취합해 가장 큰 것 하나** |
| `phase_share` | 그 비중. 어느 정도로 치우쳤는지 함께 본다 |
| `indoor` | `안`\|`밖`\|`null` — **확실할 때만**. 수영·등산 등 |
| `count` | 처방 횟수 |
| `factor_main` · `factor_lift` | 주 요인과 쏠림배수 (AI-14 §5.4) |
| `is_bundle` | 묶음 이름(`루틴`·`프로그램`)인가 |

`merge_reason` 과 `raw_forms` 를 빼지 않는다 — 합친 근거가 없으면 되돌릴 수 없다.

#### ② `exercise_targets.csv` — 주 추천 대상

| 열 | 무엇 |
|---|---|
| `exercise_id` | ① 의 id |
| `age_group` · `sex` | 처방 칸의 축 |
| `item_code` · `grade` | **과목별 측정등급.** `stats/criteria`·`grade` 가 낸다 (AI-14 §5.5) |
| `count` · `ratio` | 그 칸에서 이 운동이 나온 횟수와 비율 |

`prescription_chunks.csv` 를 **대체하지 않는다.** 저쪽은 문장이고 이쪽은 빈도다.
**칸당 30개로 자르지 않는다** — 자르면 빈도가 왜곡된다 (지금 `MAX_EXERCISES=30`).

**첫 판이 `data/release/prescription_cells.csv` 로 나 있다** (`make prescription-cells
DATA_DIR=...`) — 칸(나이×성별)×단계 708행에 이름 전부와 횟수를 싣는다. **등급 축
(`item_code`·`grade`)은 아직 없다** (AI-14 §5.5). 그것이 붙으면 이 이름이 된다.

#### ③ `video_segments.csv` — 영상 구간

`video_exercises.csv` 를 대체한다. 구간의 **끝과 길이**가 새로 붙는다.

| 열 | 무엇 |
|---|---|
| `video_id` · `start_sec` · **`end_sec`** | 구간. `end_sec` 는 **다음 운동의 시작**이다 |
| `label_end_sec` | 이름표가 마지막으로 읽힌 시각 — 길이가 아니라 `gap` 을 재는 값 |
| `gap_sec` | `end_sec - label_end_sec`. **60초를 넘으면 길이를 쓰지 않는다** |
| `duration_sec` | `end_sec - start_sec` — 강도의 근거 (AI-14 §5.3) |
| `length_basis` | 길이를 무엇으로 쟀나 — `next_start` 92 · `label_end` 30 |
| `last` | 영상의 마지막 구간인가. `gap` 필터가 못 거르는 30개가 이것이다 |
| `exercise_name` | 무슨 운동인가. `exercise_id` 는 ① 이 생긴 뒤에 붙는다 |
| `age_group` | 영상의 연령 라벨 |
| `source` | `screen`\|`screen_bar`\|`title` — 어디서 읽었나 |
| `common` | 공통 준비·마무리인가 |
| `evidence_text` | 근거 문자열 |
| `chunk_id` · `citation_label` | 이 구간을 인용할 때 실을 값 |
| `url` | `video_id` 로 만든 재생 주소. 키가 아니라 딸린 값이다 |

**난 파일은 122행 · 16열이다** (`make segments`). 위 표가 그 16열 전부다.

**길이는 시작 시각 사이로 잰다** (AI-14 §5.2). 이름표가 사라진 시점으로 재면 운동이
이어지는데도 짧게 나온다. `label.py` 의 `Segment.end_sec` 는 `gap` 을 재려고 흘려보낸다.
**`gap > 60초` 인 28개는 길이를 쓰지 않는다** — 빈 구간이 운동보다 길다(중앙 283초).

#### ④ `video_labels.csv` — 백엔드 적재 형식

**`exercise_videos` 표의 열 이름을 그대로 쓴다** (AI-14 §3.1). 라벨링 중간 산출물은
`interim/video_labeling.csv` 로 이름이 바뀌었으므로(다른 갈래) 이 이름은 **백엔드가 받는
모양 하나만** 쓴다.

```
video_id, title, channel_name, channel_type, duration_sec,
age_from, age_to, factors, intensity, space, noise, equipment,
labeled_by, label_model, collected_at
```

**이것은 적재 시점의 모양이다.** ①~③·⑤ 를 이 열에 맞춰 줄이지 않는다 — 여기서
접을 뿐이다. 영상 1편 = 1행이라 한 영상의 운동 여럿이 한 줄로 접히는데, 그 손실을
없애는 것이 **인터페이스 확정 항목**이다 (AI-14 §3.2 ㉤).

#### ⑤ `missions.csv` — 미션

**운동·영상을 id 로 잇고 라벨을 붙인다.** 값을 여기에 복사해 넣지 않는다 — 운동 이름과
구간은 ①·③ 에 있고, 여기는 잇는 자리다.

| 열 | 무엇 |
|---|---|
| `mission_id` | `{exercise_id}@{video_id}#{start_sec}` — 결정적 |
| `exercise_id` | → ① |
| `video_id` · `start_sec` | → ③ (한 구간을 가리킨다) |
| `targets` | **대상 라벨.** `연령대-성별-항목-등급` 목록. 세미콜론 |
| `phase` | ① 에서 |
| `indoor` | ① · ④ 에서 (`안`\|`밖`\|`null`) |
| `intensity` | ③ 의 `duration_sec` 삼등분 |
| `week` | `true`(주간) \| `false`(일일) |
| `common` | 공통 준비·마무리인가 — 순위를 낮춘다 |

**영상은 언제나 `video_id` 로 가리킨다.** `url` 은 ③ 에서 만들어 딸려 나가는 값이지
키가 아니다.

**대상은 여럿이다.** 한 운동이 여러 칸에 처방되므로 목록을 그대로 들고, 좁히는 것은
추천할 때 한다.

### 1.3 CSV 규약은 그대로다

`docs/02` §4 — 한 단계 = 한 CSV · 평평하게 · 다중값은 세미콜론 · `utf-8-sig` ·
원본 문자열 보존. **미션 표도 이 규약을 따른다.** 고칠 것이 없다.

---

## 2. 문서 재편

### 2.1 새로 쓰는 것 — `docs/05-코치-설계.md`

**`docs/04` 를 고쳐 쓰지 않고 새 문서를 만든다.** 미션은 청크가 아니라 표의 행이라
청킹·임베딩·임계값 규약이 걸리지 않는다. 한 문서에 두 단위를 담으면 어느 규칙이
어디에 걸리는지가 흐려진다.

담을 것 — 미션의 정의 · `mission_id` 결정성 · 라벨 값 집합(백엔드 코드값) ·
연령 폭을 정하는 규칙 · 일일/주간을 가르는 규칙 · 공통 준비·마무리 처리.

> **났다.** `AI-11` §4·§5.1 에 흩어져 있던 규칙(일일·주간 · 배분 · 연령 폭)을 옮기고
> 그쪽은 `docs/05` 를 가리키게 두었다. **규칙은 `docs/`, 실행 방법은 `docs/05`** 다.

### 2.2 범위를 좁히는 것 — `docs/04`

| 절 | 무엇을 한다 |
|---|---|
| §1 코퍼스 구성 | `source` 표에서 미션이 청크가 **아님**을 명시. 5종은 그대로 |
| §2 청크 | 첫 줄에 "**`coach/messages` 의 코퍼스 규약이다**" 를 박는다 |
| §3 검색 | 같다. **연령 필터는 여기서 손대지 않는다** — §2.4 |
| §5 재색인 | 그대로 |

### 2.3 손대는 것

| 어디 | 무엇을 |
|---|---|
| `docs/01` §6 저장소 | 코드 구역 트리에 `mission/` 을 넣는다 (§3) |
| `docs/01` §3.1 노드 | `retrieve` 가 "운동 이름 → 영상 표"에서 "라벨로 미션 고르기"로 바뀐다 |
| `docs/02` §2.3 | **고쳤다** — 채널 합집합 519편 · 라벨 상한 61편 · 유소년 신규 0편 |
| `docs/02` §4 | 미션 표 5개를 산출물 목록에 넣는다 |
| `docs/05/README.md` | §1 구역 표에 `mission/` · §2 에 AI-14 · §3 묶음을 미션 기준으로 |
| `docs/04` | 어휘 정제(①)와 빈도표(②)가 이 갈래의 산출로 붙는다 |
| `docs/02` | `end_sec` 흘려보내기 · 프레임 라벨 |
| `docs/05` | 미션 편성이 최근접 매칭에서 **라벨 고르기**로 바뀐다 |

### 2.4 연령 필터 — 문서를 고치기 전에 읽을 것

`docs/04` §3 은 연령 필터를 **안전 규약**으로 둔다. 바꾸려면 두 가지를 먼저 안다.

1. **백엔드가 이미 강제한다.** `VideoLabel.suitableFor()` 는 라벨 연령이 비면 아이에게
   내보내지 않는다. 우리가 끄고 말고 할 것이 아니다 (AI-14 §3.2)
2. 우리가 내는 **대상 라벨**이 그 필터의 입력이 된다 — 적재할 때 `age_from`~`age_to` 로 편다

**그래서 `docs/04` §3 의 문장은 그대로 두고, 미션의 연령 폭을 정하는 규칙은
`docs/05` §4 에 둔다** (§2.1). 안전 검사를 푸는 것이 아니라 다른 단위에 다른
규칙을 두는 것이다.

### 2.5 접는 것

| 문서 | 왜 |
|---|---|
| `AI-9` (`videos/search`) | 구현이 없고 백엔드가 `exercise_videos` 를 직접 조회한다 |
| `AI-4` (`assessment`) | 코드는 있으나 백엔드가 백분위·등급을 자체 계산한다 |

**지우지 않는다.** 머리에 "**보류 — 소비자가 없다. 생기면 그대로 동작한다**" 한 줄을
넣고 `README` §2 에서 보류로 표시한다. 계약은 `docs/03` 에 남는다.

---

## 3. 코드 구역

**`mission/` 을 새로 만든다.** `rag/` 는 청크·검색이고 `labeling/` 은 영상인데,
미션은 둘을 합쳐 표로 내는 일이라 어느 쪽에도 속하지 않는다.

```
mission/
  exercises.py   어휘 정제 → exercises.csv            (§1.2 ①)
  targets.py     처방 빈도표 → exercise_targets.csv    (§1.2 ②)
  segments.py    영상 구간 → video_segments.csv        (§1.2 ③)
  labels.py      프레임 라벨 → space·noise·equipment    (§1.2 ④)
  build.py       미션 표 → missions.csv                (§1.2 ⑤)
```

**옮기지 않고 그대로 두는 것** — `stats/` · `ingest/` · `api/` · `common/`.
`rag/` 는 `coach/messages` 용으로 남는다.

**`graph/` 는 비어 있다** (`__init__.py` 한 줄뿐). 미션 편성이 들어갈 자리라
AI-11 을 다시 쓸 때 채운다.

**`labeling/llm.py` 는 `src` 안에 호출자가 0이다.** 프레임 라벨(§1.2 ④)이 첫 소비자가
된다 — 지우지 말고 거기에 잇는다.

---

## 4. 순서

각 단계는 **산출물이 나와야** 다음으로 간다. 앞 단계 산출물 없이 다음을 시작하지 않는다.

| # | 무엇 | 산출 | 끝난 줄 아는 법 |
|---|---|---|---|
| ① | 어휘 정제 | `exercises.csv` | 합친 것마다 `merge_reason` 이 있다 · 갈라야 할 쌍이 안 합쳐졌다 |
| ② | 처방 빈도표 | `exercise_targets.csv` | 칸 합이 원자료 처방 횟수와 맞는다 |
| ③ | 요인 | ① 의 열이 채워짐 | **영상 라벨이 있는 10편에서 온 요인만 채워지고 나머지는 빈칸** (AI-14 §5.4) |
| ④ | 영상 구간 | `video_segments.csv` | 길이 중앙 74초(`gap ≤ 60` 인 94개). **강도는 뒤로 미룬다** — `gap` 필터가 마지막 구간을 못 걸러낸다 (AI-14 §5.2) |
| ⑤ | 프레임 라벨 | ④ 에 `space`·`noise`·`equipment` | **먼저 5~8편만.** 정확도를 보이고 전체로 |
| ⑥ | 미션 표 | `missions.csv` · `video_labels.csv` | **영상 없는 운동도 행이 된다** (`docs/03` §5.7) · `mission_id` 가 두 번 돌려도 같다 |
| ⑦ | 문서 | `docs/05` · 고친 절들 | 지운 §번호를 가리키는 링크가 0개 |

**①~④ 는 원자료와 저장소 산출물만 쓴다.** 새 API 호출도 영상 다운로드도 없다.

---

## 5. 이사 중에 깨지는 것

재편은 참조를 끊는다. 아래를 **작업 중에 확인한다.**

| 무엇 | 왜 |
|---|---|
| 문서끼리의 `§번호` 참조 | 절을 옮기면 끊긴다. `docs/` 를 grep 해서 센다 |
| `chunk_id` 대응 | 청킹 규칙을 바꾸면 저장된 인용이 끊긴다 (`docs/04` §5) |
| `exercise_vocabulary.csv` | `docs/03` §4.1 이 참조하는 닫힌 집합이다. **줄이면 계약이 깨진다** |
| 어휘 개수가 이미 어긋나 있다 | `docs/03`:361 은 **638개**, 파일은 **641개**다. 정제 전에 `AI-13` 에 올린다 |
| 같은 이름 두 파일 | 중간 산출물을 `video_labeling.csv` 로 바꿨다 (다른 갈래). `video_labels.csv` 는 적재 형식 하나만 쓴다 |
| `data/manifest.csv` | 원자료 목록(25행). 원자료를 더 넓게 읽어도 목록은 같다 |

---

## 6. 하지 않는 것

- **시설 정보** — 원자료를 받은 적이 없다. 만들지 않는다
- **`rag/` 삭제** — `coach/messages` 가 쓴다
- **`docs/03` 수정** — 팀 합의다. 어긋나면 `AI-13` 에 올린다
- **`exercise_vocabulary.csv` 축소** — 계약이 참조한다. 정제 결과는 `exercises.csv` 로 따로 낸다
- **미결 번호 재배열** — 해소된 행만 지우고 번호는 비워 둔다 (`AGENTS.md` §4)
