# 하이라이트 클립 평가

> 모델 nvidia/nemotron-3-nano-omni-30b-a3b-reasoning · 프롬프트 video-scoreboard-v3 · 추론 끔 · 점수판 전용 판독 · 해설 Whisper · `python scripts/eval_clips.py`로 재생성. 정답은 영상 제목 기준(사람 확인 전).

| 지표 | 값 |
|---|---|
| 클립 | 28개 |
| 판정 정답 (영상 판정이 정답 코드를 냄) | 15/24 (62%) |
| 오판 (정답이 아닌 판정 코드를 냄) | 2/24 (8%) |
| 단서 포착 (VLM 관찰에 정답 키워드) | 16/28 (57%) |
| 점수판을 하나도 못 읽은 클립 | 5개 |

| 리그 | 판정 | 단서 |
|---|---|---|
| KBO | 15/24 (62%) | 16/28 (57%) |

| 판정 코드 | 정답 |
|---|---|
| balk | 2/3 (67%) |
| caught_stealing | 1/1 (100%) |
| double_play | 0/3 (0%) |
| dropped_third_strike | 0/2 (0%) |
| hbp | 1/2 (50%) |
| homerun | 2/2 (100%) |
| sac_fly | 1/2 (50%) |
| steal | 3/4 (75%) |
| strikeout | 1/1 (100%) |
| triple_play | 4/4 (100%) |

## 클립별

| 클립 | 정답 | 판정 | 낸 코드 | 단서 | 점수판 판독 |
|---|---|---|---|---|---|
| kbo_balk_walkoff_bae | balk | ✓ | balk | 보크 | 2 |
| kbo_balk_ks2017 | balk | ✗ | - | ✗ | 0 |
| kbo_balk_cuevas | balk | ✓ | balk | 보크 | 0 |
| kbo_lddp_park | double_play | ✗ | - | ✗ | 0 |
| kbo_lddp_austin | double_play | ✗ | - | ✗ | 2 |
| kbo_lddp_son | double_play | ✗ | - | ✗ | 0 |
| kbo_foultip_k_walkoff | strikeout | ✓ | out, strikeout | ✗ | 2 |
| kbo_dts_walkoff | dropped_third_strike | ✗ | - | ✗ | 2 |
| kbo_dts_wp_hong | dropped_third_strike | ✗ | - | ✗ | 1 |
| kbo_homesteal_lee | steal | ✓ | out, steal | ✗ | 2 |
| kbo_homesteal_hwang | steal | ✓ | steal | ✗ | 2 |
| kbo_doublesteal_kt | steal | ✗ | out, sac_fly | 도루 | 2 |
| kbo_doublesteal_japan | steal | ✓ | pitcher, steal | ✗ | 1 |
| kbo_pickoff_ahn | caught_stealing | ✓ | caught_stealing | 견제 | 1 |
| kbo_tp_kiwoom | triple_play | ✓ | triple_play | 트리플 | 2 |
| kbo_tp_kt | triple_play | ✓ | out, triple_play | 트리플 | 2 |
| kbo_tp_lg | triple_play | ✓ | reach, triple_play | 트리플 | 2 |
| kbo_tp_lotte | triple_play | ✓ | triple_play | 트리플 | 2 |
| kbo_hbp_walkoff_ssg | hbp | ✗ | - | ✗ | 1 |
| kbo_headshot_kim | hbp | ✓ | pitcher, hbp | 헤드샷 | 2 |
| kbo_sf_walkoff_nc | sac_fly | ✗ | dropped_third_strike | 희생플라이 | 2 |
| kbo_sf_walkoff_lg | sac_fly | ✓ | reach, sac_fly | ✗ | 2 |
| kbo_hr_kim | homerun | ✓ | out, homerun | 홈런, home run | 2 |
| kbo_hr_slam_hilliard | homerun | ✓ | pitcher, homerun | 홈런, 만루 | 2 |
| kbo_squeeze_kia | (단서만) | · | reach, strikeout | 번트 | 2 |
| kbo_checkswing_review | (단서만) | · | strikeout | 체크, 판독, replay | 2 |
| kbo_abs_ejection | (단서만) | · | pitcher, strikeout | 퇴장, ABS | 0 |
| kbo_home_assist | (단서만) | · | out, triple_play | 홈 송구, 아웃, throw | 2 |
