# 처음 보는 경기 — 2025 한국시리즈 3차전 하이라이트(2025 MBC 왼쪽 위 상자, 해설 있음, 밤 경기)

> 영상 id는 -pkmgXjxquM(첫 글자 '-'가 명령 옵션으로 읽혀 파일·스냅샷 이름은 x-pkmgXjxquM). 엔진 결과를 보기 전에 투구 화면 46칸 정답을 매겼다(`../labels.json`; 점수판 확대·해설·프레임으로 확인, 정답 점검 1칸 — 8회말 '투수 교체' → 리베라토 아웃). `mbc_postseason_box`를 그대로 썼다(받아쓰기 99조각, 관절 투구 46개). **처음 측정(main `dbf4f25` 그대로, 판독 96장): 채점 42칸 중 37(88%)**, 헛판정 0, 해설 없이 35, 관절 없이 34, 점수판만 29. 판독 시각을 기록하게 한 판독기로 다시 읽었다(판독 96장). **아래 표는 이 경기를 보고 고친 뒤(이제 조정용)**. 남은 틀림(볼카운트를 점수로 읽은 판독이 적시타를 앞 칸으로 당김, 병살 해설 없음 등)은 `data/analysis/LOG.md` 2026-11-18.

다시 만들기: `python scripts/eval_games.py --snapshots ../clips_private/ks325_final --only x-pkmgXjxquM --ablation --out data/game_eval/ks325`

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| x-pkmgXjxquM (시험용 · 2025 KBO 한국시리즈 3차전 LG-한화(대전) 하이라이트 · TVING(MBC 중계)) | 42 | 38 | 3 | 1 | 0/3 |
| 합계 | 42 | 38 | 3 | 1 | 0/3 |

확실한 정답만: 40칸 중 맞음 36, 틀림 3, 놓침 1.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 안타·출루 | 14 | 13 | 1 | 0 |
| 아웃(인플레이) | 8 | 8 | 0 | 0 |
| 아웃(종류 모름) | 3 | 3 | 0 | 0 |
| 삼진 | 6 | 5 | 1 | 0 |
| 볼넷 | 5 | 5 | 0 | 0 |
| 몸에 맞는 공 | 2 | 2 | 0 | 0 |
| 홈런 | 1 | 1 | 0 | 0 |
| 병살 | 3 | 1 | 1 | 1 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | (없음) | 1 |
| double_play | double_play | 1 |
| double_play | out out | 1 |
| hbp | hbp | 2 |
| homerun | homerun out | 1 |
| out | out | 7 |
| out | out out | 1 |
| out_any | out | 2 |
| out_any | strikeout | 1 |
| reach | out | 1 |
| reach | reach | 6 |
| reach | reach balk | 1 |
| reach | reach out | 2 |
| reach | reach reach | 3 |
| reach | reach runner | 1 |
| strikeout | out | 1 |
| strikeout | strikeout | 5 |
| walk | walk | 5 |

## 입력별

| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |
|---|---|---|---|---|
| 점수판+관절+해설 | 38 | 3 | 1 | 0 |
| 점수판+관절(해설 없이) | 36 | 5 | 1 | 0 |
| 점수판+해설(관절 없이) | 34 | 6 | 2 | 0 |
| 점수판만 | 29 | 11 | 2 | 0 |
