# HANDOFF — 이어서 작업하기

> 갱신 2026-10-04. 다른 PC(맥 등)에서 `git pull` 후 이 문서부터 읽는다. 기능 설명·API는 [README.md](README.md).

## 1. 지금 상태

- main: PR #1~#24 머지, CI(test 3.9·3.12 + docker) 통과, pytest 216 · ruff clean
- **코드로 할 수 있는 건 끝났다. 다음은 실제 경기 영상 실측.** (§5)

## 2. 구조 — 누가 무엇을 하나

| 역할 | 담당 | 코드 |
|---|---|---|
| 플레이 판정 (보크·도루·아웃·득점…) | **영상**: 점수판 판독 × 장면 단서 × 해설 키워드 | `app/domain/scorebug.py`, `video_judge.py` |
| 선수 이름·구종·구속·기록·팀 맥락 | **데이터**(네이버) | `app/domain/enrich.py`, `adapters/relay/naver.py` |
| 정확도 채점 | 데이터의 플레이 결과 — **화면엔 안 씀** | `app/domain/grading.py`, `data/relay_truth/` |
| 카드·한 줄 요약·챗봇·퀴즈 | 위 결과를 기존 엔진이 그대로 사용 | `app/services/` |

- VLM: NVIDIA 호스팅 `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (VSS 3.2 Omni, 영상+음성). GPU 불필요
- LLM: Claude Sonnet 5(카드·챗봇·퀴즈), Haiku 4.5(한 줄 요약)
- 원칙: "VLM이 보고, 규칙 엔진이 판정한다." 확신이 모자라면 카드를 띄우지 않는다

## 3. 처음 세팅 (맥)

```bash
git clone https://github.com/Inje-Capstone/capstone-AI.git && cd capstone-AI
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
brew install ffmpeg                       # 영상 분석에 필요
cp .env.example .env                      # 키는 git에 없다 — 직접 채운다
.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check .
```

`.env`에 넣을 것 (값은 개인 보관 — 절대 커밋 금지):
- `NVIDIA_API_KEY` — build.nvidia.com
- `ANTHROPIC_API_KEY` + `ANTHROPIC_WORKSPACE_ID` — 지금 키는 워크스페이스 미지정이라 ID가 없으면 400
- `ROOKIE_LIVE_TOKEN` — 실시간 모드에서만 (임의 문자열, 서버·분석기 같은 값)

## 4. 파이프라인 (명령)

**리플레이 (경기 끝난 영상)**
```bash
.venv/bin/python scripts/import_naver_relay.py <네이버ID>                 # 데이터(선수·구종·채점 정답)
.venv/bin/python scripts/analyze_video.py <네이버ID> game.mp4 --limit 3   # 먼저 3분만 시험
.venv/bin/python scripts/analyze_video.py <네이버ID> game.mp4             # 전체 (재실행하면 이어서)
.venv/bin/python scripts/build_video_feed.py <네이버ID>                   # 영상 판정 + 데이터 → data/fixtures/video_<ID>.json
.venv/bin/python scripts/grade_video.py <네이버ID>                        # 정확도(규칙별 재현율·정밀도)
.venv/bin/python scripts/build_cards.py <네이버ID> && .venv/bin/python scripts/build_moments.py <네이버ID>  # 시연용 사전 생성
```

**실시간**
```bash
.venv/bin/python -m uvicorn app.main:app                 # 서버 (.env에 ROOKIE_LIVE_TOKEN)
.venv/bin/python scripts/live_video.py LIVE1 game.mp4 --away 한화 --home LG --naver-id <네이버ID>
```
5초 조각 + VLM 추론 끔 → 지연 약 5~10초. 새 상황은 `GET /api/live/LIVE1/stream`(SSE).

**점검**
```bash
.venv/bin/python scripts/simulate_video_judge.py         # 판정 로직 상한 (완벽 판독 가정) — 실경기 3개 38/38
.venv/bin/python scripts/eval_outputs.py 20260823LGOB    # LLM 출력 규칙 검사
```

## 5. 다음 할 일 (순서대로)

1. **경기 영상 확보** → `analyze_video.py --limit 3`으로 점수판·장면·해설 판독 품질 확인 → `app/adapters/video/base.py`의 프롬프트 조정
2. 같은 경기를 **추론 켬/끔**(`--no-think`)으로 각각 분석 → `grade_video.py`로 정확도 비교 → 실시간 기본값 확정
3. 판정 규칙·확신도 조정(`video_judge.py`) → 시연 경기 카드·요약 사전 생성 → 커밋
4. 백엔드 compose에 ai-engine 추가(아래 §6)
5. (생중계 소스가 생기면) 영상↔데이터 시간차 초기값을 네이버 투구 시각으로 — 지금은 장면 단서로 약 14분 뒤 고정

## 6. 외부 작업 (사람 손 필요)

- **경기 영상(mp4)**: 지도교수 승인 · 비공개 보관 · 필요한 구간만 (저작권)
- **백엔드 팀 합의**: 백엔드 `docker-compose.yml`에 ai-engine 서비스(빌드 컨텍스트 = 이 레포 git URL) + EC2 `.env`에 키. 프론트 → 백엔드 → AI 호출(AI는 내부망만). 퀴즈 포인트·팝업 조건은 백엔드 소유. 명세: `docs/openapi.json`

## 7. 결정 기록

| 날짜 | 결정 | 이유 |
|---|---|---|
| 09-27 | VSS 풀스택 대신 VSS 계열 VLM 호스팅 API | GPU·서버 부담(기술 조사서 권고) |
| 10-02 | 플레이는 영상, 선수·구종·기록은 데이터, 데이터 결과는 채점만 | "영상 분석"이 실제 판정을 하게 + 정확도를 숫자로 |
| 10-02 | GPU는 무료 API가 막히거나 품질·저작권 문제 시에만, 분석할 때만 켬 | 상시 가동 월 180만 원 |
| 10-02 | 실시간: 5초 조각 + VLM 추론 끔 | 응답 실측 1.5~2.2초(켜면 3.6~65초) |
| 10-02 | AI 비용은 캡스톤 지원금으로 처리 안 함 | 해외·프로그램·구독 금지(국고). 위장 세금계산서는 허위 증빙 |

## 8. 함정 (실측으로 알게 된 것)

- NVIDIA 호스팅: `cosmos-reason2-8b`는 이 계정에서 404, `cosmos3-nano-reasoner`는 미호스팅. 붐비면 503 → 자동 재시도(5/15/30초)
- Anthropic: Haiku 4.5는 `effort` 미지원(코드에서 자동 제외). 근거 없이 쓰면 규칙을 뒤집어 설명 → 카드·챗봇에 용어 사전 정의 주입함
- `docs/openapi.json`: 라우트를 바꾸면 `scripts/export_openapi.py` 실행(테스트가 불일치를 잡는다)
- 팀원 맥은 Python 3.9 — 3.10+ 문법 금지(CI가 3.9도 돌린다)
- 커밋 금지: `.env`, 영상 파일, `data/video/*.json`(영상 분석 결과). 네이버 데이터·LLM 스냅샷은 커밋 허용
- 정확도 수치는 실제 경기 영상 실측 전까지 약속하지 않는다(지금 숫자는 판정 로직의 상한)
