# 1080p 경기 하이라이트 채점

> `python scripts/eval_games.py --snapshots <analyze_video 결과 폴더> [--ablation]`로 다시 만든다(조정용 3경기는 ../clips_private/game_eval_final). 정답은 `labels.json`(사람이 투구 화면마다 매김), 칸별 결과는 `plays.csv`.

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| 3QYUP4AgNpI (2026 KBO 한화-삼성(대구) 하이라이트 · SBS Sports) | 36 | 34 | 2 | 0 | 0/5 |
| T93FGIYBM8E (2026 KBO 삼성-LG(잠실) 하이라이트 · MBC 스포츠+) | 47 | 37 | 5 | 5 | 2/4 |
| wyC8BnovUUA (2026 KBO 플레이오프 1차전 KIA-삼성(대구) 하이라이트 · TVING) | 35 | 28 | 3 | 4 | 0/4 |
| 합계 | 118 | 99 | 10 | 9 | 2/13 |

확실한 정답만: 100칸 중 맞음 83, 틀림 8, 놓침 9.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 안타·출루 | 35 | 31 | 3 | 1 |
| 아웃(인플레이) | 38 | 35 | 1 | 2 |
| 아웃(종류 모름) | 21 | 17 | 0 | 4 |
| 삼진 | 2 | 2 | 0 | 0 |
| 볼넷 | 7 | 5 | 1 | 1 |
| 몸에 맞는 공 | 2 | 1 | 0 | 1 |
| 홈런 | 6 | 4 | 2 | 0 |
| 희생플라이 | 2 | 1 | 1 | 0 |
| 병살 | 3 | 2 | 1 | 0 |
| 도루 | 2 | 1 | 1 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | double_play | 2 |
| double_play | out out | 1 |
| hbp | (없음) | 1 |
| hbp | hbp | 1 |
| homerun | homerun | 2 |
| homerun | homerun out | 2 |
| homerun | out | 2 |
| out | (없음) | 2 |
| out | dive_out | 1 |
| out | out | 29 |
| out | out balk | 1 |
| out | out out | 2 |
| out | out runner | 1 |
| out | out walk | 1 |
| out | strikeout | 1 |
| out_any | (없음) | 4 |
| out_any | out | 7 |
| out_any | out reach | 1 |
| out_any | strikeout | 8 |
| out_any | strikeout reach reach | 1 |
| reach | (없음) | 1 |
| reach | out | 3 |
| reach | reach | 26 |
| reach | reach out | 3 |
| reach | reach reach | 2 |
| sac_fly | out | 1 |
| sac_fly | sac_fly | 1 |
| steal | reach | 1 |
| steal | steal | 1 |
| strikeout | strikeout | 2 |
| walk | (없음) | 1 |
| walk | hbp | 1 |
| walk | walk | 4 |
| walk | walk out | 1 |

## 입력별

| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |
|---|---|---|---|---|
| 점수판+관절+해설 | 99 | 10 | 9 | 2 |
| 점수판+관절(해설 없이) | 95 | 14 | 9 | 2 |
| 점수판+해설(관절 없이) | 90 | 16 | 12 | 3 |
| 점수판만 | 89 | 17 | 12 | 3 |
