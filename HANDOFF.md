# HANDOFF — 이어서 작업하기

> 갱신 2026-10-27. 다른 PC(맥 등)에서 `git pull` 후 이 문서부터 읽는다. 기능 설명·API는 [README.md](README.md).

## 1. 지금 상태

- main: PR #1~#28 머지, CI(test 3.9·3.12 + docker) 통과, pytest 229 · ruff clean
- **판단 신호 — 관절 슬라이딩·태그·충돌(2026-10-27)**: `app/domain/contact.py` — 화면 속 모든 사람 관절(rtmlib, `PeoplePoseExtractor`)로 슬라이딩·다이빙(상체가 눕혀진 채 몸길이 0.4배 넘게 이동), 미끄러져 들어와 닿음(태그), 충돌(빠르게 다가와 닿고 넘어짐). 중계 카메라 패닝·줌은 배경 광류 + RANSAC 닮음 변환(`camera_motion`)으로 되돌린다. 스냅샷 `pose` → 장면 단서 `slide`(VLM과 같은 종류)·`slide_tag`·`collision`, 카드 근거 '[관절] …'. 충돌·태그는 홈 충돌·슬라이딩 규정·주루/수비방해 선언의 확인 장면
  - 정답 표 `data/clip_eval/contact_truth.json` 28클립(기준 6·검증 12·최종 10 — 최종은 2차 기준 고정 뒤 검출을 가리고 분류)
  - 2차: 슬라이딩 19/20 · 재현율 19/29, 최종 묶음 재현율 1/6(느린 화면 다이빙을 놓침), 태그 5/9(주자 뒤 심판 오인)
  - 3차 다듬기(2026-10-27): 슬라이딩 **24/24 · 재현율 24/29(83%)**, 태그 **9/10 · 9/11(82%)**
    - 슬라이딩: 달려와(1키/초 넘게) 몸을 던짐(눕기 1초 전 자리에서 몸길이 1.2배 — 전력질주로 추적이 끊기면 바로 앞 추적을 이어 봄), 느린 화면(1.5초 넘게 누운 채 0.25배)
    - 태그: 서 있는 사람 손목이 누운 주자에게(키의 0.3배 안, 닿고 1초 안) — 주자 뒤 심판 제외. 서서 빠르게 부딪힌 뒤 쓰러지면 충돌. 한 슬라이딩에 태그 한 번
    - 합치기: 같은 번호 / 같은 때 같은 자리(부분 검출) / 앞이 끝난 자리에서 키가 이어짐(디졸브 전환이면 크기가 튀어 따로)
    - 독립 리뷰 2회. 합성 장면에서 맞던 수정 중 실제 중계에서 손해인 것(손목 못 보면 태그 인정, 마지막 0.5초 빼기)은 되돌리고 한계로 기록
    - 기준값은 기준·검증 값으로 정하고 최종으로 확인 — 28클립을 모두 본 뒤라 새 클립으로 다시 재야 한다
  - 남은 한계: 넓은 화면 작은 다이빙·느린 화면 다이빙 일부, 같은 주자를 둘로 잡은 태그 1건, 손목이 가려진 태그, 허리 숙여 손이 무릎에 있는 심판(역할 구분 필요)
  - **처음 보는 검증(2026-10-27)**: 이미 받아 둔 MLB 28클립을 3차 기준 고정 상태로 채점(정답은 검출을 가린 후보 띠로 작성) → 슬라이딩 14/17 · 14/23(61%), 태그 4/6 · 4/8. KBO(맞춘 데이터)보다 낮은 이 수치가 일반화 성능
    - 놓친 9개 중 6개는 상체를 세운 발 먼저 슬라이딩(2루) — 다리로 누움을 보면 재현율 73→81%지만 정밀도 93→78%라 버림
    - 고친 것(holdout을 쓴 뒤): 몸 날린 수비의 태그(둘 다 누워도 손목), 슬라이딩이 끼면 충돌이 아니라 태그, 던지기 직전까지 선 모습이 보여야 '몸 던짐' → 14/16 · 14/23, 태그 5/8 · 5/8
    - 선수 역할(색)·심판 동작 자동 인식은 시도했으나 실패(색만으로 안 갈림, 자세만으로 뽑은 동작 후보 16개 중 심판 0) — 사람 분류 모델이 필요
    - 가벼운 관절 모델 비교(KBO 28클립): 슬라이딩 14/19 · 14/29, 태그 3/6 · 3/11 (기본 23/23 · 23/29, 9/10 · 9/11) — 쓰지 않는다. 시간은 기본이 영상 길이의 약 2배(관절 1.8 + 카메라 0.2), 가벼운 것 0.8배(30초 구간 단독 실측)
    - 발 먼저 슬라이딩 재도전 2가지(다리로 누움 보기, 달려 들어갈 때만 다리 보기)도 정밀도만 떨어뜨려 접음 — 베이스 위치나 자세 분류 모델이 필요
  - 클립 판정 17/27 그대로(관절 단서는 판정을 바꾸지 않고 근거·확인에 쓰임). `python scripts/contact_check.py <영상들> --truth data/clip_eval/contact_truth.json`, `eval_clips.py --pose`, `analyze_video.py --pose`(10분 구간씩). CPU로 영상 길이의 3~4배 걸린다
- **해설만 vs 화면만 vs 둘 다(2026-10-27, 발표 근거)**: 같은 KBO 27클립·같은 판정 규칙에 입력만 바꿈 → 둘 다 17/27(63%) · 해설만 10/27(37%) · 화면만 7/27(26%, VLM 장면 설명 글은 소리를 들었을 수 있어 뺌). 화면이 더한 9개는 해설만으로는 못 가린 클립(홈스틸 2·더블스틸·견제사·희생플라이·홈런·보크·파울팁 삼진·비디오 판독). 화면 오독(VLM이 지어낸 포수 놓침·점수판 아웃 오독) 때문에 놓친 2개도 있다. `python scripts/eval_clips.py --clips ../clips_private --report-only --scorebug --asr --ablation` → [ABLATION.md](data/clip_eval/ABLATION.md)
- **판단 신호 실험 — VLM 장면 v4, 채택 안 함(2026-10-26)**: 선수 충돌·공이 사람에 맞음·공이 경기장 밖으로·바운드 담장·관중 접촉·장비 날아감을 VLM 장면 종류로 추가해 KBO 28클립을 다시 돌림 → 판정 17→14, 오판 4→7. 새 신호는 세리머니를 '충돌', 물 뿌리기를 '장비 날아감'으로 적는 등 대부분 잡음이고, 종류 목록이 길어지자 기존 단서까지 흔들렸다. 지시문 v3 유지. 판단 신호는 VLM 말고 관절·물체 검출(CV)로 따로 재야 한다. 결과 비공개 폴더 `results_v4/`
- **심판 선언 인식(2026-10-26)**: 타격방해·주루방해·수비방해·비디오 판독·체크스윙·퇴장 등 19종(`video_judge.CALL_TABLE`) — 해설·자막 낱말 + 확인(심판 동작·판독 화면·자막·60초 안 반복)일 때만 call 이벤트(점수판 상태는 안 바꿈) → 카드(`detectors.CALL_CARDS`). 새 판정은 표에 한 줄 + 카드 한 줄 + 장면 테스트 한 줄
- **영상 분석 4단계(2026-10-26)**: KBO 하이라이트 판정 17/27 · 오판 4/27(심판 선언 클립 3개에 정답 추가)(`data/clip_eval/REPORT.md`). 보크 클립 18개 중 14개에서 보크 판정(해설+자세) — 못 잡은 4개는 해설 없는 직캠·음악 영상(자세만, 확신 0.35라 안 띄움). 투수 자세: 다리 듦 직전 투구 시작 움직임(≤0.5초)은 정지에서 빼지 않고, 무릎 기준선 오르내림은 한 투구, 축발은 그 투구에서 든 다리의 반대쪽
- **행동 판정 로직(2026-10-26)**: 영상 대상 규칙 233개마다 성립 조건·관측 신호·판정(`scripts/action_logic_src.py`) — 인식(심판 판정 알아보기) 가능 152 · 판단(동작 직접 보기) 가능 9. 미구현 신호 우선순위는 RULES_COVERAGE.md. 재검수 검사(단계·보이는 방법·영상 코드·동작 판정표 정합)와 규칙 장면 테스트(`tests/test_rule_scenarios.py`)가 어긋나면 실패
- **판정 로직 재점검(2026-10-26)**: 모든 영상 판정 코드에 근거 조항·성립 조건을 붙여 저장 규칙과 대조 — [RULES_COVERAGE.md](RULES_COVERAGE.md) '영상 판정 로직', '보크·투구 반칙 — 동작 판정 방법'. 코드와 표가 어긋나면 빌드 실패
- **투수 자세 보크 판정(실험)**: `scripts/balk_check.py` (선택 설치 `[pose]`) — 해설 + 동작으로 판정, 점수판은 검증. 정확도 실측 전
- **하이라이트 클립 평가(KBO 24개)**: 영상 판정 0% → **46%** (`data/clip_eval/REPORT.md`). 점수판 전용 판독 + Whisper 해설 받아쓰기 + 지시문 v3. **MVP는 KBO만** — MLB는 MVP 이후
- **야구 규칙 전체 목록 302개**(`data/rules_catalog.json`)와 엔진 커버리지 보고서 [RULES_COVERAGE.md](RULES_COVERAGE.md) — 공식 야구규칙(OBR 2024) 721조항 100% 매핑(`data/obr_provisions.json`, 번호만 — 빠지면 테스트 실패) · 설명 100% · 영상 결과 인식 15%. 규칙·영상을 새로 받으면 목록부터 고친다(§5-0)
- 온보딩 진단 퀴즈·개인화 가중치를 백엔드 DTO 모양으로 제공: `/api/backend/onboarding/*` (README "백엔드 온보딩 연동")
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

**하이라이트 클립 평가** (영상은 레포 밖 비공개 폴더, 정답지 `data/clip_eval/labels.json`)
```bash
.venv/bin/pip install -e ".[asr]"                       # 해설 받아쓰기(Whisper, gRPC)
.venv/bin/python scripts/eval_clips.py --clips ../clips_private --scorebug --asr   # KBO만(기본)
.venv/bin/python scripts/eval_clips.py --clips ../clips_private --scorebug --asr --report-only  # 캐시로 재채점(무료)
```
- 점수판: `app/adapters/video/scorebug_reader.py` — 모서리를 잘라 3배 키워 한 장씩 읽고, 언제 읽을지는 로컬 썸네일 비교로 정한다
- 해설: `app/adapters/video/speech_asr.py` — NVIDIA 호스팅 Whisper large-v3, 같은 NVIDIA_API_KEY

**점검**
```bash
.venv/bin/python scripts/simulate_video_judge.py         # 판정 로직 상한 (완벽 판독 가정) — 실경기 3개 38/38
.venv/bin/python scripts/eval_outputs.py 20260823LGOB    # LLM 출력 규칙 검사
```

## 4-1. 영상 판독 비용 단계 — 무료 API → GPU 직접

**1단계 (지금): NVIDIA 무료 호스팅 API** — `.env`의 `NVIDIA_API_KEY`만 있으면 된다.
- 크레딧: 가입 시 1,000 (프로필에서 "Request More"로 최대 5,000 요청). 회사 메일이면 90일 AI Enterprise 체험 + 4,000 추가. 분당 40회 제한
- 다 쓰면 자동 충전은 없다 → 2단계로
- 소모 추정(요청 1회 = 1크레딧 가정): 리플레이 3시간 경기 60초 조각 ≈ 180회 / **실시간 5초 조각 ≈ 2,160회 → 1,000크레딧으로 실시간 한 경기를 다 못 돈다**

**2단계: GPU를 분석할 때만 빌려 직접 띄운다** — 코드 수정 없음, 주소만 바꾼다.
- 모델: 같은 모델(Nemotron 3 Nano Omni) 가중치가 Hugging Face에 공개돼 있다. 라이선스 NVIDIA Open Model Agreement — 상업 이용까지 허용
  - NVIDIA NIM 컨테이너(`nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`)도 있지만 별도 평가 라이선스. 개발자 프로그램 회원은 연구·개발·테스트 용도로 GPU 16장까지 무료 → 캡스톤은 해당. **기본은 vLLM + HF 가중치**(로그인 불필요, FP8이 L40S에 들어감)
- GPU: FP8(약 33GB) = **L40S 48GB 1장** / BF16(약 62GB) = H100·A100 80GB 1장. 디스크 70GB 이상
- 요금(2026-10 조사, 바뀜): 해외 RunPod L40S 시간당 약 $0.8~1.1 · H100 약 $2~3 / 국내 엘리스클라우드 H100 GPU당 ₩4,270 · A100 80GB ₩1,970
  - 국내 업체 종량제는 지원금 규정(해외·구독 금지)에 안 걸릴 수도 있다 — 학교 담당자 확인 필요
- 경기당 추정(실측 전): 리플레이 = 모델 받기 10~20분 + 분석 수십 분 → L40S로 $1 안팎. 실시간 = 경기 3시간 내내 켜야 함

```bash
# GPU 서버 (RunPod이면 템플릿 이미지 vllm/vllm-openai:v0.20.0)
VLM_API_KEY=<아무 긴 문자열> bash scripts/gpu_serve.sh          # FP8 · L40S
MODEL_VARIANT=BF16 VLM_API_KEY=<...> bash scripts/gpu_serve.sh  # H100·A100 80GB

# 일반 GPU 서버(docker)라면
docker run --gpus all --ipc host -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $PWD/scripts:/s -e VLM_API_KEY=<...> --entrypoint bash vllm/vllm-openai:v0.20.0 /s/gpu_serve.sh

# 분석 PC — .env의 키 대신 위에서 정한 문자열을 넘긴다(환경변수가 .env보다 우선)
NVIDIA_API_KEY=<같은 문자열> .venv/bin/python scripts/analyze_video.py <ID> game.mp4 --base-url http://<서버IP>:8000/v1 --limit 3
NVIDIA_API_KEY=<같은 문자열> .venv/bin/python scripts/live_video.py LIVE1 game.mp4 --away 한화 --home LG --base-url http://<서버IP>:8000/v1
```

- 첫 실행은 가중치 다운로드로 오래 걸린다. 디스크(볼륨)를 남겨 두면 다음엔 바로 뜬다(볼륨 보관비만 듦)
- **끝나면 GPU(Pod)를 반드시 정지** — 켜 둔 시간만큼 과금
- 이 절차는 아직 실제 GPU로 돌려 보지 않았다. 첫 실행 때 `--limit 3`으로 응답 형식부터 확인

## 5. 다음 할 일 (순서대로)

**5-0. 규칙·영상이 새로 오면** — `scripts/rules_catalog_src.py`에 한 줄 추가(조항·행동→판정·보이는 방법·단계) → 용어 사전 근거가 없으면 `data/glossary_seed.json`에 추가 → `python scripts/build_rules_catalog.py` (테스트가 불일치를 잡는다)

**영상 판정 넓히기 로드맵** (단계는 RULES_COVERAGE.md 기준)
- 2단계 72개 = 심판이 이미 내린 판정을 '알아보기'. 학습 없이 프롬프트·규칙으로 한다: 장면 단서에 심판 동작(세이프·아웃·페어·파울·타임·보크 지적)·판독 화면·자막 문구를 추가하고, `video_judge`에 인정 2루타·타격방해·자동 고의4구·포구 판독 같은 코드를 늘린다
- 평가셋: 클립 목록(규칙별 525개)에 카탈로그 id로 정답을 달아 규칙별 재현율을 잰다 — 이게 있어야 '얼마나 맞히나'를 말할 수 있다
- 3단계 77개 = AI가 동작을 보고 반칙을 직접 '판단'(보크 정지·체크스윙·3피트·슬라이딩·태그 타이밍 등). 여기서부터 영상 학습(파인튜닝·자세 추정). 유형마다 라벨 클립 수백 개가 필요하니 시연 가치가 큰 것부터 고른다
- 4단계 30개(기록·행정)는 영상 판정 대상이 아니다 — 질문이 오면 용어 사전 근거로 설명만


0. **경기 전체 분석(analyze_video)에도 점수판 전용 판독·Whisper 해설을 붙인다** — 지금은 클립 평가에만 연결돼 있다
1. **경기 영상 확보** → `analyze_video.py --limit 3`으로 점수판·장면·해설 판독 품질 확인 → `app/adapters/video/base.py`의 프롬프트 조정
   - **한국어 해설을 알아듣는지 먼저 확인** — 모델 안내에 언어 지원이 영어로만 적혀 있다(§8). 안 되면 `--no-audio`로 점수판·장면만 쓰거나 한국어 음성인식을 따로 붙인다
2. 같은 경기를 **추론 켬/끔**(`--no-think`)으로 각각 분석 → `grade_video.py`로 정확도 비교 → 실시간 기본값 확정
3. 판정 규칙·확신도 조정(`video_judge.py`) → 시연 경기 카드·요약 사전 생성 → 커밋
4. 백엔드 compose에 ai-engine 추가(아래 §6)
5. (생중계 소스가 생기면) 영상↔데이터 시간차 초기값을 네이버 투구 시각으로 — 지금은 장면 단서로 약 14분 뒤 고정

## 6. 외부 작업 (사람 손 필요)

- **경기 영상(mp4)**: 지도교수 승인 · 비공개 보관 · 필요한 구간만 (저작권)
- **백엔드 팀 합의**: 백엔드 `docker-compose.yml`에 ai-engine 서비스(빌드 컨텍스트 = 이 레포 git URL) + EC2 `.env`에 키. 프론트 → 백엔드 → AI 호출(AI는 내부망만). 퀴즈 포인트·팝업 조건은 백엔드 소유. 명세: `docs/openapi.json`
- **백엔드 온보딩**: 진단은 `POST /api/backend/onboarding/diagnosis` 호출 → `learningLevel` 저장. AI 문항은 보기 4개 → 백엔드 `optionId @Max(3)`을 4로 풀거나 AI 문항(`/questions`)을 그대로 쓰기

## 7. 결정 기록

| 날짜 | 결정 | 이유 |
|---|---|---|
| 09-27 | VSS 풀스택 대신 VSS 계열 VLM 호스팅 API | GPU·서버 부담(기술 조사서 권고) |
| 10-02 | 플레이는 영상, 선수·구종·기록은 데이터, 데이터 결과는 채점만 | "영상 분석"이 실제 판정을 하게 + 정확도를 숫자로 |
| 10-02 | GPU는 무료 API가 막히거나 품질·저작권 문제 시에만, 분석할 때만 켬 | 상시 가동 월 180만 원 |
| 10-02 | 실시간: 5초 조각 + VLM 추론 끔 | 응답 실측 1.5~2.2초(켜면 3.6~65초) |
| 10-10 | 무료 크레딧 소진 후엔 GPU 1장(L40S)에 vLLM + 공개 가중치(FP8)로 직접 띄움 | 같은 모델이라 코드 수정 없음, 상업 이용 허용 라이선스, 경기당 $1 안팎 추정 |
| 10-02 | AI 비용은 캡스톤 지원금으로 처리 안 함 | 해외·프로그램·구독 금지(국고). 위장 세금계산서는 허위 증빙 |

## 8. 함정 (실측으로 알게 된 것)

- **영상 모델의 언어 (실측 2026-10-26)**: Nemotron Omni는 한국어 해설을 못 알아듣고 지어낸 문장·영어 문장을 적는다 → 해설은 Whisper로 받는다. 지시문에 키워드 목록을 주면 듣지 않은 단어를 그대로 적어(가짜 단서) v3에서 뺐다
- **영상 통째 판독은 점수판에 약하다**: 같은 프레임도 점수판만 잘라 키워 한 장으로 주면 정확히 읽는다. 여러 장을 한 요청에 묶으면 다시 틀린다
- **하이라이트의 한계**: 점수판이 몇 초만 비치고, 직캠은 점수판·해설이 없다 — 판정 상한이 낮다. 실제 경기 영상(중계 전체)이 MVP 평가의 기준
- 영상 모델 입력 한도: 영상 1개 최대 2분(1080p는 1fps·128프레임까지). 지금 조각(60초·5초)은 안에 든다
- NVIDIA 호스팅: `cosmos-reason2-8b`는 이 계정에서 404, `cosmos3-nano-reasoner`는 미호스팅. 붐비면 503 → 자동 재시도(5/15/30초)
- Anthropic: Haiku 4.5는 `effort` 미지원(코드에서 자동 제외). 근거 없이 쓰면 규칙을 뒤집어 설명 → 카드·챗봇에 용어 사전 정의 주입함
- `docs/openapi.json`: 라우트를 바꾸면 `scripts/export_openapi.py` 실행(테스트가 불일치를 잡는다)
- 팀원 맥은 Python 3.9 — 3.10+ 문법 금지(CI가 3.9도 돌린다)
- 커밋 금지: `.env`, 영상 파일, `data/video/*.json`(영상 분석 결과). 네이버 데이터·LLM 스냅샷은 커밋 허용
- 정확도 수치는 실제 경기 영상 실측 전까지 약속하지 않는다(지금 숫자는 판정 로직의 상한)
