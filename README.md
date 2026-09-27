# 루키(ROOKIE) AI 엔진

야구 중계를 인지해 **입문자 눈높이의 상황 설명**을 만들어 내는 Python/FastAPI 서비스.
프론트(React)·백엔드와 HTTP로만 붙는 독립 서비스다.

> ⚠️ **현재 중계·기록 데이터는 실데이터가 아니라 손으로 만든 fixture다.**
> 실소스(네이버/KBO 문자중계, KBO 기록·STATIZ)가 확보되면 `app/adapters/` 구현체만 교체한다.
> 화면·보고서에 노출될 때는 카드의 `source` 필드로 시연 여부를 밝힌다.

## 빠른 시작

```bash
cd capstone-AI
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# API 키 없이도 전부 돈다 (mock 백엔드)
.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check .
.venv/bin/python -m uvicorn app.main:app --reload
```

> Windows는 `.venv/bin/` 대신 `.venv/Scripts/`를 쓴다.

API 문서: http://localhost:8000/docs · 헬스체크: http://localhost:8000/health

실제 모델을 쓰려면 `.env.example`을 `.env`로 복사하고 `ANTHROPIC_API_KEY`를 채운다.
키는 **서버에만** 둔다 — 클라이언트/번들에 절대 넣지 않는다.

## 검증 명령

```bash
.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check .
```

완료 선언·커밋 전 필수 통과. API 키가 없어도 전량 통과하도록 설계했다(mock LLM).
GitHub Actions(`.github/workflows/ci.yml`)가 `main`·`develop` push/PR마다 같은 명령과 Docker 빌드를 돌린다.

## Docker

```bash
docker build -t rookie-ai-engine .
docker run --rm -p 8000:8000 --env-file .env rookie-ai-engine
```

키는 이미지에 굽지 않고 실행 시 `--env-file`(또는 compose `environment`)로 넣는다.
배포 시 `ROOKIE_CORS_ORIGINS`에 프론트 도메인을 쉼표로 적는다(기본은 로컬 개발 서버 5173·3000).

## API (와이어프레임 화면과 1:1)

| 엔드포인트 | 화면 |
|---|---|
| `GET /api/games` | S3 홈 경기 리스트 (`has_video`로 비활성 처리) |
| `GET /api/games/{id}/state?t=` | S4 좌하단 스코어보드 |
| `GET /api/games/{id}/cards?t=&level=&category=` | S4 우상단 AI 설명 카드 스택 |
| `POST /api/games/{id}/cards/{card_id}/simplify` | S4 "더 쉽게 설명" |
| `GET /api/games/{id}/matchup?t=` | S4 우하단 선수·매치업 분석 |
| `POST /api/games/{id}/chat` | S4 하단 챗봇 |

`level`은 `입문 | 초보 | 익숙`, `category`는 반복 쿼리 파라미터(`기본 룰`, `구종 · 투구`,
`전술 · 기록`, `응원 문화`). 온보딩 답변을 그대로 넘기면 된다 — 유저 DB는 백엔드 팀 소유다.

```bash
curl 'localhost:8000/api/games/20260823LGOB/cards?t=7550&level=입문' --get
```

## 구조

```
app/
  domain/        순수 로직 (I/O 없음) — 여기만 보면 "무엇을 어떻게 판단하는지"가 다 보인다
    game_state.py  중계 이벤트 → 경기 상태(이닝·주자·카운트·스코어)
    detectors.py   특이 상황 감지 규칙 테이블
    profile.py     온보딩 답변 → 노출 가중치
    timeline.py    영상 타임코드 ↔ 중계 타임코드
  adapters/      외부 격리 — relay(중계) / stats(기록) / llm(Claude·mock)
  services/      카드 생성, 챗봇, 분석 패널
  api/           FastAPI 라우터 = 프론트 계약
  prompts/       프롬프트 원문(.md)
data/
  fixtures/      가상 경기 중계
  stats/         가상 선수 기록
  glossary_seed/ 용어 사전 시드 (data/glossary_seed.json)
  snapshots/     배치 사전 생성 결과 (gitignore — 필요할 때 다시 만든다)
```

### 설계에서 중요한 것 네 가지

1. **카드는 타임코드의 결정적 함수다.** `cards(t)`는 그 시점까지 감지된 상황을 프로필로 거른 결과다.
   시킹·배속이 그냥 되고, 미리 만들어 두면 데모 중 LLM 호출이 0이 된다.
2. **분기 대신 파라미터.** 난이도 3 × 관심 4 = 12조합에 if문을 두지 않는다.
   규칙마다 `중요도 × 관심도 × 난이도배율`을 계산해 임계값으로 거른다(`domain/profile.py`).
3. **모든 카드에 근거(`reasons`)가 붙는다.** 왜 감지됐고 왜 이 사람에게 보이는지를 숫자까지 남긴다.
   근거를 못 대면 설계가 틀린 것이다.
4. **외부가 죽어도 화면은 산다.** 스냅샷 → 실시간 생성 → 생략 순으로 내려간다.
   LLM이 완전히 죽어도 스코어보드와 기록 패널은 그대로 뜬다.

## 배치 사전 생성

```bash
.venv/bin/python scripts/build_cards.py 20260823LGOB --levels 0 1 2
```

경기 전체 카드를 미리 만들어 `data/snapshots/{game_id}.json`에 굳힌다.
실제 모델이면 **Batch API(표준가 50%)** 를 쓰고, 스냅샷에는 완성 카드뿐 아니라
**원본 중계 이벤트도 함께** 저장한다 — 프롬프트를 고치면 같은 재료로 전량 재생성할 수 있다.

> mock으로 만든 스냅샷은 실제 모델로 도는 환경에서 **자동으로 무시된다**.
> 조립 문장이 생성물인 척 나가는 걸 막기 위해서다.

## LLM 백엔드

`ROOKIE_LLM_BACKEND` 환경변수로 강제할 수 있다.

| 값 | 동작 |
|---|---|
| `auto` (기본) | 키가 있으면 `claude`, 없으면 `mock` |
| `claude` | Claude Sonnet 5 (`ROOKIE_LLM_MODEL`로 변경) |
| `mock` | 용어 사전 시드를 조립. **생성이 아니다** — 테스트·오프라인 데모용 |
| `fail` | 항상 실패. 폴백 경로를 실제로 밟아볼 때 |

## 아직 안 한 것

- 실제 문자중계 크롤링 (`adapters/relay/` 에 자리만 비워둠)
- 실제 선수 기록 소스 (`adapters/stats/`)
- 중계 음성 STT 기반 영상 동기화 — 지금은 fixture에 타임코드가 박혀 있다
- 용어 사전 본문 콘텐츠 (지금은 난이도별 한 줄 시드만)
- 라이브 모드(캡스톤2)
