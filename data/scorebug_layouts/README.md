# 점수판 위치 파일

중계사마다 점수판 모양이 다르다. 위치 파일 하나에 칸 위치(화면 비율 0~1)를 적어 두면, 주자·아웃·초말·삼진 표시는 화소로 읽고 VLM은 점수·볼카운트·회차 숫자만 읽는다. 화소가 VLM보다 정확했다(주자·아웃 VLM 0~1/6 → 화소 7/7, `../analysis/LOG.md` 2026-10-28).

## 파일

| 파일 | 중계 | 잰 영상 | 확인 |
|---|---|---|---|
| `sbs_sports_2026.json` | SBS Sports 2026 두 줄 가로 띠 | 3QYUP4AgNpI | 5/5 |
| `bar_two_line.json` | 같은 SBS 띠를 따로 잰 것(좌표 0.001 안으로 같음) | p55O9v1xKRI | 2/2 |
| `mbc_sports_plus.json` | MBC 스포츠+ 아래 띠(흰 주자 칸) | T93FGIYBM8E | 6/6, 초말 4/4 |
| `tving_postseason.json` | 2026 포스트시즌 TVING 아래 띠 | wyC8BnovUUA | |
| `kbo_topleft_box.json` | 왼쪽 위 상자형 | zRqpvJdTybA | 10/10, 초말 5/5 |
| `kbo_box_bottomright.json` | 같은 상자형이 오른쪽 아래(KBS N) | NVOfYTDrqlU | 2/2 |
| `kbo_caster_box.json` | SPOTV 캐스터 이름 띠가 붙은 상자 | YiiGl9UUV2E | 2/2 |
| `kbo_caster_box_up.json` | 같은 판, 위 띠가 얇아 30px 위 | xtXEisvgTR0 | 9/9 |
| `spotv2_box.json` | SPOTV2(B·S·O 점 줄) | 89gNJQjBRrQ | 2/2, 초말 4/4 |
| `kbs_postseason_box.json` | KBS N 포스트시즌 왼쪽 위 상자(볼·스트라이크·아웃 모두 점, ▲/▼ 하나, 보라 'nK') — crop은 제목 줄 뺌(숫자가 회차 오독을 부름) | p-BBD0DMTYw, IT2iLKT3a6c | 5/5·3/3, 초말 5/5·3/3, 카운트 5/5·3/3, 표시 2/2 |
| `mbc_postseason_box.json` | MBC 포스트시즌 왼쪽 위 상자(제목 줄 · 양옆 팀 칸 · 가운데 다이아몬드, 볼카운트는 숫자, 아웃 점 2개, 검은 ▲/▼) — crop은 제목 줄 뺌 | H8ydwy0nGOI | 17/17, 초말 15/15 |
| `sbs_sports.json` | SBS 2024(중간발표 시연 클립) | | |

## 칸

| 키 | 뜻 | 필수 |
|---|---|---|
| `name` | 이름 | ✓ |
| `bases` | 1·2·3루 칸 중심 `{"1": [x, y], …}` | ✓ |
| `outs` | 아웃 점 중심들 | ✓ |
| `crop` | 점수판 영역 `[x, y, 너비, 높이]` — VLM에 이 영역만 보낸다 | 권장 |
| `scale` | VLM에 보낼 때 키우는 배율(기본 3, 넓은 띠는 1.5) | |
| `base_fill` | 주자 칸 채우는 방식 `color`(색) · `bright`(흰색, MBC) | |
| `base_hue` | 주자 칸 색상 범위(도) — 빨간 삼진 글자를 주자로 읽지 않게 | |
| `half` | ▲/▼ 칸 `[x0, y0, x1, y1]` | |
| `half_fill` | 켜진 화살표 색 `color` · `bright` · `dark`(검정, SPOTV2) | |
| `half_by` | `shape`(화살표 하나의 모양) · `position`(▲·▼ 둘 다 그리고 켜진 쪽만 칠할 때) | |
| `balls`·`strikes` | 볼·스트라이크를 점으로 그리는 판의 점 중심들 — VLM이 점을 못 센다 | |
| `mark` | 삼진 표시('nK')가 덮는 칸 `[x0, y0, x1, y1]` | |
| `mark_fill`·`mark_hue` | 표시 글자 `color`(+색상 범위, 기본 빨강) · `dark`(검정) | |
| `cover_val` | 아웃 점이 모두 이보다 밝은 무채색이면 표시가 덮은 것(0.85) | |
| `panel` | 점수판 바탕 고정 색 `[x, y, r, g, b]` 2~3곳 — 있으면 영상을 화소로 훑어 표시를 찾는다 | |
| `prompt_note` | VLM 판독 지시에 덧붙이는 이 점수판만의 안내(영어) — 볼카운트를 'X-Y'로 그려 점수와 헷갈리는 판 | |

## 새 중계사 재는 법

1. 점수판이 보이는 장면을 크게 잘라 격자를 그려 본다(주자 있는 장면·아웃 있는 장면 포함).
2. 칸 중심을 1080p 화소로 재 화면 비율로 적는다. 3루는 2루를 기준으로 1루와 대칭.
3. 정답을 아는 장면으로 확인: `python scripts/scorebug_layout.py game.mp4 data/scorebug_layouts/<이름>.json 30:-:1 310:1:1` (시각:주자:아웃).
4. 삼진 표시가 있는 중계면 `mark`를, 표시가 판독 사이에 뜨면 `panel`도 잰다. 바탕 지점은 점수판이 보일 때 색이 채널마다 0.03 안으로 일정하고, 없는 장면(관중·잔디)과 0.1 넘게 다른 곳 — 팀 색 칸은 팀마다 달라 쓰지 않는다.
5. 같은 중계라도 위치가 달라질 수 있다(SPOTV 상자는 위 띠 두께에 따라 30px) — 영상마다 장면 두 개로 먼저 확인한다.
