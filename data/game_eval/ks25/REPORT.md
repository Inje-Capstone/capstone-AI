# 처음 보는 경기 — 2025 한국시리즈 1차전 하이라이트(KBS 왼쪽 위 상자, 해설 있음, 낮 경기)

> 엔진 결과를 보기 전에 투구 화면 18칸 정답을 매겼다(`../labels.json`; 해설·점수판 확대로 확인). 2025 KBS 상자라 `kbs_postseason_box`를 그대로 썼다. 판독기 v6으로 처음부터 읽었다(판독 103장, 받아쓰기 87조각, 관절 투구 18개 — 투구 화면 찾기가 정지 화면 묶음을 기준으로 골라 투구 화면을 18개만 찾았다). **처음 측정(main `3fc233a` 그대로): 채점 17칸 중 15(88%)**, 헛판정 0, 해설 없이 13, 관절 없이 15, 점수판만 13. **아래 표는 이 경기를 보고 고친 뒤(이제 조정용)** — 해설의 '스윙/루킹 스트라이크 아웃'을 삼진으로, 받아쓰기 '본래 출루'를 '볼넷'으로. 기록은 `data/analysis/LOG.md` 2026-11-17. 다시 만들기: `python scripts/eval_games.py --snapshots ../clips_private/ks25_final --only X_eXehIWgZ8 --ablation --out data/game_eval/ks25`

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| X_eXehIWgZ8 (시험용 · 2025 KBO 한국시리즈 1차전 한화-LG(잠실) 하이라이트 · TVING(KBS 중계)) | 17 | 17 | 0 | 0 | 0/0 |
| 합계 | 17 | 17 | 0 | 0 | 0/0 |

확실한 정답만: 17칸 중 맞음 17, 틀림 0, 놓침 0.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 안타·출루 | 8 | 8 | 0 | 0 |
| 아웃(인플레이) | 3 | 3 | 0 | 0 |
| 삼진 | 5 | 5 | 0 | 0 |
| 볼넷 | 1 | 1 | 0 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| out | out | 1 |
| out | out out | 1 |
| out | out runner | 1 |
| reach | out reach | 1 |
| reach | reach | 1 |
| reach | reach out | 2 |
| reach | reach out reach | 1 |
| reach | reach reach out strikeout | 1 |
| reach | reach sac_fly out hbp hbp balk hbp reach | 1 |
| reach | reach strikeout | 1 |
| strikeout | strikeout | 2 |
| strikeout | strikeout homerun out reach | 1 |
| strikeout | strikeout reach | 1 |
| strikeout | strikeout strikeout | 1 |
| walk | walk | 1 |

## 입력별

| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |
|---|---|---|---|---|
| 점수판+관절+해설 | 17 | 0 | 0 | 0 |
| 점수판+관절(해설 없이) | 13 | 4 | 0 | 0 |
| 점수판+해설(관절 없이) | 17 | 0 | 0 | 0 |
| 점수판만 | 13 | 4 | 0 | 0 |
