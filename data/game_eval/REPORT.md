# 1080p 경기 하이라이트 채점

> `python scripts/eval_games.py --snapshots ../clips_private/game_eval`로 다시 만든다. 정답은 `labels.json`(사람이 투구 화면마다 매김), 칸별 결과는 `plays.csv`.

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| 3QYUP4AgNpI (2026 KBO 한화-삼성(대구) 하이라이트 · SBS Sports) | 36 | 29 | 5 | 2 | 1/4 |
| T93FGIYBM8E (2026 KBO 삼성-LG(잠실) 하이라이트 · MBC 스포츠+) | 47 | 32 | 7 | 8 | 5/1 |
| wyC8BnovUUA (2026 KBO 플레이오프 1차전 KIA-삼성(대구) 하이라이트 · TVING) | 36 | 27 | 3 | 6 | 0/4 |
| 합계 | 119 | 88 | 15 | 16 | 6/9 |

확실한 정답만: 97칸 중 맞음 74, 틀림 9, 놓침 14.

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | double_play | 1 |
| double_play | out | 1 |
| double_play | out out out | 1 |
| hbp | hbp | 1 |
| homerun | (없음) | 2 |
| homerun | double_play | 1 |
| homerun | homerun out | 2 |
| homerun | out | 1 |
| out | (없음) | 5 |
| out | dive_out | 1 |
| out | out | 27 |
| out | out balk | 1 |
| out | out out | 2 |
| out | out walk | 1 |
| out | runner | 1 |
| out_any | (없음) | 5 |
| out_any | double_play | 1 |
| out_any | out | 6 |
| out_any | out reach | 1 |
| out_any | strikeout | 7 |
| out_any | strikeout reach reach | 1 |
| reach | (없음) | 2 |
| reach | double_play | 1 |
| reach | out | 2 |
| reach | reach | 24 |
| reach | reach out | 3 |
| reach | reach reach | 2 |
| sac_fly | out | 1 |
| sac_fly | out runner | 1 |
| steal | reach | 1 |
| steal | steal | 1 |
| strikeout | strikeout | 2 |
| walk | (없음) | 1 |
| walk | hbp | 1 |
| walk | reach | 2 |
| walk | walk | 4 |
| walk | walk out | 1 |
| walk_or_hbp | (없음) | 1 |

## 입력별

| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |
|---|---|---|---|---|
| 점수판+관절+해설 | 88 | 15 | 16 | 6 |
| 점수판+관절(해설 없이) | 88 | 15 | 16 | 6 |
| 점수판+해설(관절 없이) | 83 | 20 | 16 | 6 |
| 점수판만 | 84 | 19 | 16 | 6 |
