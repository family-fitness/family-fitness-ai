# 기여 규약 (AI 레포)

팀 공통 규약을 이 레포에 맞춰 적은 것이다.
이슈·PR 템플릿은 조직 `.github` 레포에서 공통 관리하므로 여기에는 두지 않는다.

## 처음 한 번

```bash
git config core.hooksPath .githooks     # 브랜치명·커밋 메시지 로컬 검사
git config commit.template .gitmessage  # 커밋 메시지 틀
```

훅은 파이썬 3로 돈다. 급할 때는 `--no-verify`로 넘길 수 있으나,
GitHub ruleset이 같은 정규식으로 다시 막으므로 미루는 것에 가깝다.

## 브랜치

| 브랜치 | 몫 |
|---|---|
| `main` | 항상 배포 가능한 상태. 운영 반영, 태그 관리 |
| `develop` | 다음 릴리스 통합 브랜치. 기능 브랜치가 모이는 곳 |
| `feature/*` | 단일 기능 단위. 끝나면 PR로 `develop`에 병합 |
| `hotfix/*` | `main`에서 발견된 긴급 이슈. 수정 후 `main` + `develop` 양쪽 반영 |
| `release/*` | 배포 전 최종 점검과 버전 태깅 |

이름 규칙 — `^(feature|hotfix|chore|docs|release)/[\w\-]+`

```
feature/AI-12-grade-card-percentile
hotfix/AI-4-fix-null-citation
release/v1-0-0
chore/update-dependencies
docs/api-guide
```

AI 레포의 이슈 번호 접두어는 `AI-`를 쓴다. (FE-, BE-와 구분)

## 커밋

형식 — `<type>(<scope>)?<!>?: <제목>`

`build` `chore` `ci` `docs` `feat` `fix` `perf` `refactor` `revert` `style` `test`

scope는 이 레포에서 `rag` `stats` `agent` `api` `labeling` `data` `docs` 정도를 쓴다.

```
feat: 등급 카드 백분위 계산 추가
fix(rag): 인용 누락된 청크 필터링 오류 수정
feat!: 미션 승인 게이트 응답 스키마 변경
```

제목은 무엇을 했는지 쓴다. `fix: 버그 수정`, `feat: 업데이트`는 형식만 통과할 뿐 아무 말도 하지 않는다.

## PR

1. `develop`에서 `feature/*`를 딴다.
2. 작업하고 푸시한다. 브랜치명이 규약에 어긋나면 `pre-push`가 막는다.
3. `develop`으로 PR을 연다. 템플릿은 조직 `.github`에서 자동으로 붙는다.
4. `Closes #<이슈번호>`로 이슈를 연결한다.
5. `auto-assign` 워크플로가 담당자를 자동으로 붙인다.

## 데이터·비밀값

- 원자료는 커밋하지 않는다. 재현은 수집·산출 스크립트로 한다. (`.gitignore` 참고)
- `.env`는 올리지 않는다. 키가 늘면 `.env.example`에만 이름을 추가한다.
