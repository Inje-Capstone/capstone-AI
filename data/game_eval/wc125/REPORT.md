# 처음 보는 경기 — 2025 와일드카드 결정전 1차전 하이라이트(2025 SBS 왼쪽 위 상자, 해설 있음, 낮 경기)

> 엔진 결과를 보기 전에 투구 화면 28칸 정답을 매겼다(`../labels.json`; 해설·점수판 확대·프레임으로 확인, 정답 점검 1칸 — 6회말 병살 → 2루 포스아웃). `sbs_postseason_2025`를 그대로 썼다(받아쓰기 113조각, 관절 투구 28개). **처음 측정(main `dbf4f25` 그대로): 채점 27칸 중 24(89%)**, 헛판정 0, 해설 없이 24, 관절 없이 24, 점수판만 24. 판독 시각을 기록하게 한 판독기로 다시 읽었다(판독 149장). **아래 표는 이 경기를 보고 고친 뒤(이제 조정용)** — 앞당긴 판독의 늦은 카운트, 새 반 이닝 투구를 덮는 해설, 받아쓰기 '스윙 선진'. 기록은 `data/analysis/LOG.md` 2026-11-18.

다시 만들기: `python scripts/eval_games.py --snapshots ../clips_private/wc125_final --only LAbgb6yDMMw --ablation --out data/game_eval/wc125`

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| LAbgb6yDMMw (시험용 · 2025 KBO 와일드카드 결정전 1차전 NC-삼성(대구) 하이라이트 · TVING(SBS 중계)) | 27 | 27 | 0 | 0 | 0/1 |
| 합계 | 27 | 27 | 0 | 0 | 0/1 |

확실한 정답만: 25칸 중 맞음 25, 틀림 0, 놓침 0.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 안타·출루 | 9 | 9 | 0 | 0 |
| 아웃(인플레이) | 8 | 8 | 0 | 0 |
| 아웃(종류 모름) | 2 | 2 | 0 | 0 |
| 삼진 | 3 | 3 | 0 | 0 |
| 볼넷 | 1 | 1 | 0 | 0 |
| 볼넷·사구 | 1 | 1 | 0 | 0 |
| 홈런 | 2 | 2 | 0 | 0 |
| 병살 | 1 | 1 | 0 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | double_play walk balk | 1 |
| homerun | homerun | 1 |
| homerun | homerun out out reach | 1 |
| out | out | 5 |
| out | out balk | 1 |
| out | out out | 1 |
| out | out walk out runner out strikeout | 1 |
| out_any | strikeout | 1 |
| out_any | strikeout out out | 1 |
| reach | out reach out out runner walk | 1 |
| reach | reach | 4 |
| reach | reach double_play | 1 |
| reach | reach out | 1 |
| reach | reach reach out runner reach | 1 |
| reach | reach strikeout out out reach out balk | 1 |
| strikeout | strikeout | 3 |
| walk | walk out strikeout | 1 |
| walk_or_hbp | hbp | 1 |

## 입력별

| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |
|---|---|---|---|---|
| 점수판+관절+해설 | 27 | 0 | 0 | 0 |
| 점수판+관절(해설 없이) | 25 | 2 | 0 | 0 |
| 점수판+해설(관절 없이) | 24 | 3 | 0 | 0 |
| 점수판만 | 24 | 3 | 0 | 0 |
