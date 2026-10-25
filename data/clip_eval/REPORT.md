# 하이라이트 클립 평가

> 모델 nvidia/nemotron-3-nano-omni-30b-a3b-reasoning · 프롬프트 video-scoreboard-v2 · 추론 끔 · `python scripts/eval_clips.py`로 재생성. 정답은 영상 제목 기준(사람 확인 전).

| 지표 | 값 |
|---|---|
| 클립 | 53개 |
| 판정 정답 (영상 판정이 정답 코드를 냄) | 6/45 (13%) |
| 단서 포착 (VLM 관찰에 정답 키워드) | 28/53 (53%) |
| 점수판을 하나도 못 읽은 클립 | 7개 |

| 리그 | 판정 | 단서 |
|---|---|---|
| KBO | 3/22 (14%) | 13/26 (50%) |
| MLB | 3/23 (13%) | 15/27 (56%) |

| 판정 코드 | 정답 |
|---|---|
| balk | 2/7 (29%) |
| double_play | 0/5 (0%) |
| dropped_third_strike | 0/2 (0%) |
| hbp | 0/3 (0%) |
| homerun | 0/3 (0%) |
| infield_fly | 0/2 (0%) |
| passed_ball | 0/2 (0%) |
| sac_fly | 3/3 (100%) |
| steal | 0/6 (0%) |
| strikeout | 1/4 (25%) |
| triple_play | 0/4 (0%) |
| walk | 0/3 (0%) |
| wild_pitch | 0/1 (0%) |

## 클립별

| 클립 | 정답 | 판정 | 낸 코드 | 단서 | 점수판 판독 |
|---|---|---|---|---|---|
| kbo_balk_walkoff_bae | balk | ✓ | balk | ✗ | 2 |
| kbo_balk_ks2017 | balk | ✗ | runner | 보크 | 8 |
| kbo_balk_cuevas | balk | ✗ | - | 보크 | 1 |
| mlb_balk_walkoff_cle | balk | ✗ | reach, double_play | 보크 | 3 |
| mlb_balk_walkoff_mil | balk | ✗ | pitcher, reach | 보크 | 2 |
| mlb_balk_walkoff_cws | balk | ✓ | balk | 보크, balk | 3 |
| mlb_balk_walkoff_norris | balk | ✗ | pitcher, homerun | 보크 | 2 |
| kbo_lddp_park | double_play | ✗ | - | ✗ | 0 |
| kbo_lddp_son | double_play | ✗ | - | ✗ | 0 |
| mlb_lddp_pagan | double_play | ✗ | dropped_third_strike | 병살 | 2 |
| mlb_lddp_negron | double_play | ✗ | reach | ✗ | 2 |
| mlb_dp_slide_int | double_play | ✗ | reach | 슬라이딩, slide | 4 |
| kbo_foultip_k_walkoff | strikeout | ✗ | - | ✗ | 2 |
| mlb_foultip_k | strikeout | ✓ | strikeout | ✗ | 3 |
| mlb_bunt_foul_k | strikeout | ✗ | out | ✗ | 2 |
| mlb_immaculate_kim | strikeout | ✗ | - | ✗ | 0 |
| kbo_dts_walkoff | dropped_third_strike | ✗ | reach | ✗ | 4 |
| kbo_dts_wp_hong | dropped_third_strike | ✗ | - | ✗ | 0 |
| mlb_iff_puig | infield_fly | ✗ | out, pitcher, out | 인필드 | 3 |
| mlb_iff_mets | infield_fly | ✗ | reach | 인필드 | 2 |
| kbo_homesteal_lee | steal | ✗ | dropped_third_strike, out | ✗ | 3 |
| kbo_homesteal_hwang | steal | ✗ | out, runner, runner, out | ✗ | 14 |
| kbo_doublesteal_kt | steal | ✗ | reach, homerun, out, pitcher | 도루 | 8 |
| kbo_doublesteal_japan | steal | ✗ | reach, pitcher, homerun | 도루 | 3 |
| mlb_delayed_steal_murphy | steal | ✗ | - | steal | 4 |
| mlb_delayed_steal_trout | steal | ✗ | - | 도루 | 2 |
| kbo_tp_kiwoom | triple_play | ✗ | double_play, double_play, reach | 트리플 | 4 |
| kbo_tp_kt | triple_play | ✗ | - | 트리플 | 5 |
| kbo_tp_lg | triple_play | ✗ | steal, reach | 트리플 | 3 |
| kbo_tp_lotte | triple_play | ✗ | runner | ✗ | 3 |
| kbo_hbp_walkoff_ssg | hbp | ✗ | - | ✗ | 2 |
| kbo_headshot_kim | hbp | ✗ | - | 몸에 맞 | 10 |
| mlb_hbp_ejected | hbp | ✗ | pitcher, out | 몸에 맞 | 10 |
| kbo_sf_walkoff_nc | sac_fly | ✓ | reach, out, sac_fly, pitcher | ✗ | 10 |
| kbo_sf_walkoff_lg | sac_fly | ✓ | sac_fly, pitcher | ✗ | 2 |
| mlb_sf_walkoff_phillips | sac_fly | ✓ | sac_fly | ✗ | 4 |
| kbo_hr_kim | homerun | ✗ | out, reach, pitcher, out, reach | 홈런 | 9 |
| kbo_hr_slam_hilliard | homerun | ✗ | reach, pitcher, out | 홈런 | 3 |
| mlb_ipthr_naquin | homerun | ✗ | reach | ✗ | 2 |
| mlb_walk_votto_3ball | walk | ✗ | pitcher | 볼넷 | 3 |
| mlb_walk_bryant | walk | ✗ | - | ball four | 2 |
| mlb_clock_walkoff_walk | walk | ✗ | balk | ✗ | 4 |
| mlb_pb_blanks | passed_ball | ✗ | reach | ✗ | 2 |
| mlb_pb_lofton | passed_ball | ✗ | pitcher | 포일 | 0 |
| mlb_wp_walkoff_sf | wild_pitch | ✗ | runner, reach | ✗ | 6 |
| mlb_ci_walkoff_lad | (단서만) | · | - | ✗ | 0 |
| mlb_grd_fence_gillespie | (단서만) | · | wild_pitch | ✗ | 2 |
| mlb_grd_ortiz | (단서만) | · | - | ✗ | 1 |
| mlb_obstruction_ws | (단서만) | · | double_play, reach, dropped_third_strike, out | obstruction | 5 |
| kbo_squeeze_kia | (단서만) | · | walk, out, reach, out, reach, reach, reach, out, runner | ✗ | 20 |
| kbo_checkswing_review | (단서만) | · | runner | 체크, 판독 | 5 |
| kbo_abs_ejection | (단서만) | · | - | ABS | 0 |
| kbo_home_assist | (단서만) | · | out, pitcher | 아웃 | 5 |
