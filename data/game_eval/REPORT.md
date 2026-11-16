# 1080p 경기 하이라이트 채점

> `python scripts/eval_games.py --snapshots <analyze_video 결과 폴더> [--ablation]`로 다시 만든다(조정용 3경기는 ../clips_private/game_eval_final_v5). 정답은 `labels.json`(사람이 투구 화면마다 매김), 칸별 결과는 `plays.csv`.

> 11-10: 판독 시각을 판독기 v2(덩어리 판독을 주자·아웃 화소가 같아진 뒤로)대로 다시 찍은 스냅샷 `../clips_private/game_eval_final_v5`로 채점한 값이다 — 판독 내용은 그대로, 시각만(`data/analysis/LOG.md` 2026-11-10).

> 11-15: 판독기 v4(화소 상태가 새로 이어진 구간에 판독이 없으면 한 장 더)의 추가 판독을 더한 스냅샷 `../clips_private/game_eval_final_v5`로 채점한 값이다 — 기존 판독은 그대로(`data/analysis/LOG.md` 2026-11-15).

> 11-16: 정답 점검 — T93FGIYBM8E 751.5초 '몸에 맞는 공' → '볼넷'(해설 '풀카운트 바깥쪽 빠지는군요', 직전 점수판 3-2; '몸을 숙임'은 보호대를 푸는 동작). 고치기 전 엔진은 이 칸을 사구로 냈다 — 고친 정답으로 main `6325ee5`는 112, 지금 113(`data/analysis/LOG.md` 2026-11-16).

> 11-17: 판독기 v5(주자·아웃이 바뀐 판독의 시각을 그 화소 상태가 처음 보인 초로 2초 이상 앞당김)로 시각을 다시 찍고, 위치 파일에 표시·바탕이 있으면 삼진 표시를 다시 훑은 스냅샷 `../clips_private/game_eval_final_v5`로 채점한 값이다 — 판독 내용은 그대로, VLM 요청 없음(`data/analysis/LOG.md` 2026-11-17).

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| 3QYUP4AgNpI (2026 KBO 한화-삼성(대구) 하이라이트 · SBS Sports) | 37 | 36 | 1 | 0 | 0/4 |
| T93FGIYBM8E (2026 KBO 삼성-LG(잠실) 하이라이트 · MBC 스포츠+) | 47 | 44 | 3 | 0 | 1/5 |
| wyC8BnovUUA (2026 KBO 플레이오프 1차전 KIA-삼성(대구) 하이라이트 · TVING) | 35 | 33 | 1 | 1 | 0/4 |
| 합계 | 119 | 113 | 5 | 1 | 1/13 |

확실한 정답만: 101칸 중 맞음 96, 틀림 4, 놓침 1.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 안타·출루 | 35 | 34 | 1 | 0 |
| 아웃(인플레이) | 39 | 39 | 0 | 0 |
| 아웃(종류 모름) | 21 | 20 | 0 | 1 |
| 삼진 | 2 | 2 | 0 | 0 |
| 볼넷 | 8 | 7 | 1 | 0 |
| 몸에 맞는 공 | 1 | 1 | 0 | 0 |
| 홈런 | 6 | 5 | 1 | 0 |
| 희생플라이 | 2 | 2 | 0 | 0 |
| 병살 | 3 | 2 | 1 | 0 |
| 도루 | 2 | 1 | 1 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | double_play | 2 |
| double_play | out out | 1 |
| hbp | hbp | 1 |
| homerun | homerun | 3 |
| homerun | homerun out | 2 |
| homerun | out | 1 |
| out | dive_out | 1 |
| out | out | 34 |
| out | out balk | 1 |
| out | out runner | 1 |
| out | out steal | 1 |
| out | out walk | 1 |
| out_any | (없음) | 1 |
| out_any | out | 8 |
| out_any | strikeout | 12 |
| reach | out | 1 |
| reach | reach | 30 |
| reach | reach out | 2 |
| reach | reach reach | 2 |
| sac_fly | sac_fly | 2 |
| steal | caught_stealing reach | 1 |
| steal | steal | 1 |
| strikeout | strikeout | 2 |
| walk | hbp | 1 |
| walk | walk | 6 |
| walk | walk out | 1 |

## 입력별

| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |
|---|---|---|---|---|
| 점수판+관절+해설 | 113 | 5 | 1 | 1 |
| 점수판+관절(해설 없이) | 110 | 8 | 1 | 1 |
| 점수판+해설(관절 없이) | 98 | 13 | 8 | 3 |
| 점수판만 | 97 | 14 | 8 | 3 |
