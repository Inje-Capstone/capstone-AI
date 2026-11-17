# 처음 보는 경기 — 2025 준플레이오프 4차전 하이라이트(MBC 왼쪽 위 상자, 해설 있음, 밤 경기)

> 엔진 결과를 보기 전에 투구 화면 44칸 정답을 매겼다(`../labels.json`; 해설·점수판 확대·프레임 격자로 확인). 한 해 전 중계지만 MBC 포스트시즌 상자 위치가 2026과 같아 `mbc_postseason_box`를 그대로 썼다. 판독기 v6으로 처음부터 읽었다(판독 98장, 받아쓰기 95조각, 관절 투구 17개 — 이 중계는 투구 화면 찾기가 회차 카드 묶음을 기준으로 삼아 거칠다, `data/analysis/LOG.md` 2026-11-17). **처음 측정(main `c980a7d` 그대로): 채점 27칸 중 22(81%)**, 헛판정 1, 해설 없이 19, 관절 없이 22, 점수판만 17. **아래 표는 이 경기를 보고 고친 뒤(이제 조정용)** — 체크스윙을 스윙으로 판정했다는 말('스윙입니다')을 2스트라이크 아웃의 삼진 근거로. 다시 만들기: `python scripts/eval_games.py --snapshots ../clips_private/spo25_final --only UXHXNlgF4jU --ablation --out data/game_eval/spo25`

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| UXHXNlgF4jU (시험용 · 2025 KBO 준플레이오프 4차전 SSG-삼성(대구) 하이라이트 · TVING(MBC 중계)) | 27 | 23 | 4 | 0 | 1/15 |
| 합계 | 27 | 23 | 4 | 0 | 1/15 |

확실한 정답만: 23칸 중 맞음 19, 틀림 4, 놓침 0.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 안타·출루 | 5 | 2 | 3 | 0 |
| 아웃(인플레이) | 7 | 7 | 0 | 0 |
| 아웃(종류 모름) | 1 | 1 | 0 | 0 |
| 삼진 | 8 | 7 | 1 | 0 |
| 볼넷 | 1 | 1 | 0 | 0 |
| 몸에 맞는 공 | 1 | 1 | 0 | 0 |
| 볼넷·사구 | 3 | 3 | 0 | 0 |
| 홈런 | 1 | 1 | 0 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| hbp | reach out hbp | 1 |
| homerun | out walk homerun homerun out | 1 |
| out | out | 6 |
| out | out out | 1 |
| out_any | out strikeout | 1 |
| reach | out | 2 |
| reach | reach | 1 |
| reach | reach reach | 1 |
| reach | walk | 1 |
| strikeout | out | 1 |
| strikeout | out strikeout | 1 |
| strikeout | strikeout | 3 |
| strikeout | strikeout out | 2 |
| strikeout | strikeout strikeout | 1 |
| walk | walk walk | 1 |
| walk_or_hbp | runner walk | 1 |
| walk_or_hbp | walk | 2 |

## 입력별

| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |
|---|---|---|---|---|
| 점수판+관절+해설 | 23 | 4 | 0 | 1 |
| 점수판+관절(해설 없이) | 19 | 8 | 0 | 1 |
| 점수판+해설(관절 없이) | 23 | 4 | 0 | 1 |
| 점수판만 | 17 | 10 | 0 | 1 |
