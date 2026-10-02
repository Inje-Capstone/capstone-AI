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
키가 워크스페이스에 묶여 있지 않으면 `ANTHROPIC_WORKSPACE_ID`(콘솔 Settings → Workspaces)도 넣는다 — 없으면 400.
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

### EC2 배포 (2026-09-30 결정 — 기존 EC2에서 직접 빌드)

레지스트리(ECR)를 쓰지 않고 **EC2에서 리포를 받아 빌드한다.**

```bash
git clone https://github.com/Inje-Capstone/capstone-AI.git && cd capstone-AI
docker build -t rookie-ai-engine .
docker run -d --name rookie-ai --restart unless-stopped \
  -p 127.0.0.1:8000:8000 --env-file .env rookie-ai-engine
curl localhost:8000/health
```

- **`-p 127.0.0.1:8000:8000`** — 인스턴스 내부에만 연다. 엔진에는 인증이 없으므로
  `0.0.0.0`으로 열면 누구나 챗봇을 호출할 수 있고 그 비용은 우리 API 키에서 나간다.
  같은 EC2의 Spring Boot는 `http://127.0.0.1:8000`으로 부른다.
- **아키텍처**: EC2(x86_64)에서 빌드하면 맞는다. **Apple Silicon 맥에서 빌드한 이미지는 EC2에서 뜨지 않는다**
  (필요하면 `docker build --platform linux/amd64`). EC2에서 빌드하기로 한 이유 중 하나다.
- CI가 `main`·PR마다 **linux/amd64에서 이미지 빌드를 검증**한다 — 빌드가 깨진 상태로 배포될 일은 없다.
- `--restart unless-stopped`로 인스턴스 재부팅 후 자동 기동. 로그는 `docker logs`만 쓰면 재시작 시 사라지므로
  운영에 쓸 거면 CloudWatch 등으로 뺀다.
- 런타임에 **GPU도 NVIDIA API도 쓰지 않는다** — 일반 인스턴스로 충분하다.
  NVIDIA 키는 오프라인 배치(`scripts/analyze_video.py`) 전용이다.

## API (와이어프레임 화면과 1:1)

| 엔드포인트 | 화면 |
|---|---|
| `GET /api/games` | S3 홈 경기 리스트 (`has_video`로 비활성 처리) |
| `GET /api/games/{id}/state?t=` | S4 좌하단 스코어보드 |
| `GET /api/games/{id}/cards?t=&level=&category=` | S4 우상단 AI 설명 카드 스택 |
| `POST /api/games/{id}/cards/{card_id}/simplify` | S4 "더 쉽게 설명" |
| `GET /api/games/{id}/matchup?t=` | S4 우하단 선수·매치업 분석 |
| `POST /api/games/{id}/chat` | S4 하단 챗봇 |
| `GET /api/glossary?q=&category=` | S5 용어 사전 목록·검색 |
| `GET /api/glossary/{term_id}?level=` | S5 용어 상세 (카드 `term_id` 딥링크) |
| `GET /api/games/{id}/today-rules?t=&level=&category=` | 시청 종료 팝업 — 이 경기에서 카드로 본 룰 (LLM 호출 없음) |
| `POST /api/quiz` | 오늘 본 룰 퀴즈 (하루 5문제). `term_ids`는 백엔드가 그날 본 것을 모아 넘기고, `seed`(사용자ID+날짜)로 같은 날 같은 문제 |
| `GET /api/games/{id}/moment?t=&level=` | S4 방금 장면 한 줄 요약 (Haiku 4.5, 모델 없으면 중계 원문 조립 — 항상 응답) |
| `GET /api/onboarding/diagnostic` | 온보딩 B① 수준 진단 문항 3개. **정답 비노출** · LLM 호출 없음 |
| `POST /api/onboarding/diagnostic` | 진단 채점 → 입문(0~1개) · 초보(2개) · 익숙(3개) + 결과 문구 · 문항별 해설 |

진단은 **보상 없는 자기 진단**이다(포인트는 하루 퀴즈 몫). 채점 응답은 답을 낸 문항의 정답·해설을
함께 주므로 — 결과 화면이 그걸 보여줘야 한다 — 시험처럼 신뢰할 수 있는 관문으로 쓰지 말 것.
응시 1회 제한·재응시 이력이 필요하면 상태를 가진 백엔드가 감싼다.

`level`은 `입문 | 초보 | 익숙`, `category`는 반복 쿼리 파라미터(`기본 룰`, `구종 · 투구`,
`전술 · 기록`, `응원 문화`). 온보딩 답변을 그대로 넘기면 된다 — 유저 DB는 백엔드 팀 소유다.

```bash
curl 'localhost:8000/api/games/20260823LGOB/cards?t=7550&level=입문' --get
```

## API 명세 (백엔드 연동용)

백엔드(Spring Boot)가 DTO를 생성하는 원본은 **OpenAPI 명세**다. 손으로 옮겨 적은 문서는 곧 낡는다.

```bash
.venv/bin/python scripts/export_openapi.py   # docs/openapi.json 갱신
```

- 커밋된 파일: [docs/openapi.json](docs/openapi.json) — 서버를 띄우지 않고도 바로 쓸 수 있다
- 서버가 떠 있으면 `/openapi.json`(원본) · `/docs`(대화형)도 같은 내용이다
- **계약을 바꾸고 이 스크립트를 다시 돌리지 않으면 `tests/test_openapi_export.py`가 실패한다** —
  남의 코드가 조용히 낡은 계약을 따라가는 걸 막기 위해서다. 실패하면 다시 내보내고 함께 커밋하라.

Spring Boot 쪽은 이 파일을 openapi-generator에 넣어 클라이언트·DTO를 생성하면 된다.
주의할 계약 두 가지:

- `POST /api/games/{id}/cards/{card_id}/simplify`는 **POST인데 요청 본문이 없다**(경로·쿼리만 쓴다).
- `category`는 **반복 쿼리 파라미터**다(`?category=A&category=B`). 콤마로 합치면 안 되고,
  값에 공백·가운뎃점이 있어(`구종 · 투구`) URL 인코딩이 필요하다.

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

데모 경기(`20260823LGOB`)는 실제 모델(Sonnet 5 Batch · Haiku 4.5)로 만든 카드 51장·한 줄 요약 216줄을
`data/snapshots/`에 커밋해 뒀다 — 배포 서버는 모델을 부르지 않고 바로 서빙한다.

> mock으로 만든 스냅샷은 실제 모델로 도는 환경에서 **자동으로 무시된다**.
> 조립 문장이 생성물인 척 나가는 걸 막기 위해서다.

## 영상 판정 (VSS 방식) — 영상으로 판정하고 문자중계로 검증

**서비스 화면의 상황 인지는 영상에서만 나온다.** 네이버 문자중계는 같은 경기의 정답지로 정확도를
재는 데만 쓴다. "VLM이 보고, 규칙 엔진이 판정한다."

```bash
# .env에 NVIDIA_API_KEY (build.nvidia.com) — 서버 런타임은 쓰지 않는다
.venv/bin/python scripts/analyze_video.py 20260920HHLG02026 game.mp4 --limit 3   # 앞 3분만 시험
.venv/bin/python scripts/analyze_video.py 20260920HHLG02026 game.mp4            # 전체 (재실행하면 이어서)
.venv/bin/python scripts/build_video_feed.py 20260920HHLG02026                  # 판정 → data/fixtures/video_{id}.json
.venv/bin/python scripts/grade_video.py 20260920HHLG02026                       # 문자중계로 채점
```

1. **분석**(`analyze_video.py`): 60초 청크(360p, 해설 음성 유지)를 VLM에 보내 세 가지를 받는다 —
   **점수판 판독**(이닝·볼카운트·아웃·주자·점수), **장면 단서**(슬라이딩·포수 놓침·담장 넘김…),
   **해설 키워드**("보크", "도루"…). VLM은 판정하지 않는다. `data/video/{id}.json`에 청크 단위로 누적.
2. **판정**(`app/domain/video_judge.py`): 점수판 변화의 모양 × 단서로 보크·도루·도루 실패·폭투·볼넷·사구·
   낫아웃·인필드플라이·희생플라이·병살·홈런·투수 교체를 가른다. 확신도와 근거를 남기고, 확신이 모자라면
   상태만 반영하고 종류는 붙이지 않는다(카드 안 뜸). 결과는 기존 엔진(GameSim → 감지 규칙 → 카드·챗봇·퀴즈)이
   그대로 받고, 카드 근거에 "영상 근거: 점수판 …, 해설 '보크' …"가 붙는다.
3. **검증**(`grade_video.py`): 같은 경기 문자중계(`data/relay_truth/`)와 상황 단위로 비교해 규칙별 재현율·정밀도.
   영상이 덮은 구간만 센다. 시각은 영상 오프셋 추정으로 맞춘다.

판정 로직 상한(`scripts/simulate_video_judge.py`, 실경기 중계로 "완벽한 눈" 판독·단서를 합성):
실경기 3개 38/38 일치, 단서 30% 누락 + 점수판 10% 오독에서 재현율 92%·정밀도 97%.
**실제 정확도는 경기 영상 실측으로만 말한다.** 구종(결정구 변화구 카드)은 영상 판정 범위 밖이다.

- VLM: VSS 풀스택 대신 **VSS 계열 VLM을 NVIDIA 호스팅 API로 직접** 부른다(GPU 불필요, 기술 조사서 권고안).
  기본 `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`(VSS 3.2 Omni, 영상+음성). 무료가 막히거나 품질이
  모자라면 GPU 1장을 분석할 때만 빌려 `--base-url http://host:8000/v1`로 붙인다. VSS 서버면 `--backend vss`
- ⚠️ 실제 야구 영상에서 점수판·장면 판독 품질은 실측 전. NVIDIA 트라이얼 약관상 운영 금지 — 사전 배치 분석 전용

## 정답지 가져오기 (네이버 문자중계 — 검증 전용)

```bash
.venv/bin/python scripts/import_naver_relay.py 20260920HHLG02026
```

네이버 스포츠 문자중계를 `data/relay_truth/naver_{id}.json`으로 변환한다. **서비스 화면에는 쓰지 않는다**
(`data/fixtures/`가 아니라 정답지 폴더). 2019년 이후 정규시즌 경기는 끝난 뒤에도 받을 수 있다(표본 확인).
받은 직후 재생한 최종 점수가 네이버 점수와 다르면 저장하지 않는다. 실경기 3개(2026-09-20)를 커밋해 뒀다.

## 생성 결과 검사

```bash
.venv/bin/python scripts/eval_outputs.py 20260823LGOB --report eval.json --max-fail 0.05
```

현재 백엔드로 카드·한 줄 요약·챗봇(레드팀 질문 포함)·퀴즈를 뽑아 규칙 검사를 돌린다(`app/evals.py`):
길이·문장 수·한국어 비율, **맥락에 없는 숫자**(지어낸 기록), **시스템 프롬프트 누출**, 퀴즈 형식.
프롬프트를 고친 뒤 실제 모델로 돌려 회귀를 본다. 통과가 "좋은 설명"을 뜻하진 않는다 — 나쁜 출력을 거르는 그물이다.

## Python 버전

3.9~3.12 지원(CI가 둘 다 돈다). 3.9는 이미 EOL이고 anthropic SDK 1.x는 3.10+를 요구해서,
3.9에선 0.x SDK(0.125)가 깔린다 — 지금 쓰는 파라미터는 둘 다 지원한다. 팀 환경이 3.10+로 올라가면
`requires-python`과 ruff `target-version`을 함께 올린다.

## LLM 백엔드

`ROOKIE_LLM_BACKEND` 환경변수로 강제할 수 있다.

| 값 | 동작 |
|---|---|
| `auto` (기본) | 키가 있으면 `claude`, 없으면 `mock` |
| `claude` | Claude Sonnet 5 (`ROOKIE_LLM_MODEL`로 변경). 한 줄 요약만 Haiku 4.5 (`ROOKIE_SUMMARY_MODEL`) |
| `mock` | 용어 사전 시드를 조립. **생성이 아니다** — 테스트·오프라인 데모용 |
| `fail` | 항상 실패. 폴백 경로를 실제로 밟아볼 때 |

## 아직 안 한 것

- 타자 대 투수 개인 전적 (네이버 미제공 — 실경기는 선발의 상대 팀 성적으로 대신)
- 영상 오프셋을 중계 음성(STT)으로 맞추기 — 지금은 영상 관찰을 중계 이벤트와 다수결로 짝지어
  자동 추정하고(`analyze_video.py --apply`), 표가 모자라면 `--video-offset`으로 수동 지정한다
- 용어 사전 본문 확충 (지금은 용어 14개 × 난이도별 한 줄 + 관련 용어)
- 라이브 모드(캡스톤2)
