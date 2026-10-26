# 해설만 vs 화면만 vs 둘 다

> 같은 클립·같은 판정 규칙에 입력만 바꿨다. 해설 = 중계 소리(Whisper 받아쓰기), 화면 = 점수판·장면 종류(VLM)·화면 자막 — VLM이 소리도 들었으니 화면만에선 장면 설명 글을 뺐다. `python scripts/eval_clips.py --ablation`로 재생성.

| 입력 | 판정 정답 | 오판 |
|---|---|---|
| 화면 + 해설 | 17/27 (63%) | 4/27 |
| 해설만 | 10/27 (37%) | 2/27 |
| 화면만 | 7/27 (26%) | 6/27 |

- 화면이 더한 클립 9개: kbo_balk_cuevas, kbo_foultip_k_walkoff, kbo_homesteal_lee, kbo_homesteal_hwang, kbo_doublesteal_japan, kbo_pickoff_ahn, kbo_sf_walkoff_lg, kbo_hr_kim, kbo_checkswing_review
- 화면 때문에 놓친 클립 2개: kbo_doublesteal_kt, kbo_sf_walkoff_nc

## 클립별

| 클립 | 정답 | 화면 + 해설 | 해설만 | 화면만 |
|---|---|---|---|---|
| kbo_balk_walkoff_bae | balk | ✓ | ✓ | ✗ |
| kbo_balk_ks2017 | balk | ✗ | ✗ | ✗ |
| kbo_balk_cuevas | balk | ✓ | ✗ | ✓ |
| kbo_lddp_park | double_play | ✗ | ✗ | ✗ |
| kbo_lddp_austin | double_play | ✗ | ✗ | ✗ |
| kbo_lddp_son | double_play | ✗ | ✗ | ✗ |
| kbo_foultip_k_walkoff | strikeout | ✓ | ✗ | ✗ dropped_third_strike |
| kbo_dts_walkoff | dropped_third_strike | ✗ | ✗ | ✗ |
| kbo_dts_wp_hong | dropped_third_strike | ✗ | ✗ | ✗ |
| kbo_homesteal_lee | steal | ✓ | ✗ | ✓ |
| kbo_homesteal_hwang | steal | ✓ | ✗ | ✓ |
| kbo_doublesteal_kt | steal | ✗ sac_fly | ✓ | ✗ |
| kbo_doublesteal_japan | steal | ✓ | ✗ strikeout | ✓ |
| kbo_pickoff_ahn | caught_stealing | ✓ | ✗ | ✓ |
| kbo_tp_kiwoom | triple_play | ✓ | ✓ | ✗ |
| kbo_tp_kt | triple_play | ✓ | ✓ | ✗ double_play |
| kbo_tp_lg | triple_play | ✓ | ✓ | ✗ double_play |
| kbo_tp_lotte | triple_play | ✓ | ✓ | ✗ double_play |
| kbo_hbp_walkoff_ssg | hbp | ✗ | ✗ | ✗ |
| kbo_headshot_kim | hbp | ✓ | ✓ | ✗ |
| kbo_sf_walkoff_nc | sac_fly | ✗ dropped_third_strike | ✓ | ✗ dropped_third_strike |
| kbo_sf_walkoff_lg | sac_fly | ✓ | ✗ | ✗ |
| kbo_hr_kim | homerun | ✓ | ✗ | ✓ |
| kbo_hr_slam_hilliard | homerun | ✓ | ✓ | ✓ |
| kbo_squeeze_kia | squeeze | ✗ strikeout | ✗ strikeout | ✗ |
| kbo_checkswing_review | video_review | ✓ | ✗ | ✗ check_swing,strikeout |
| kbo_abs_ejection | ejection | ✓ | ✓ | ✗ |
