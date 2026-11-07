"""투구 뒤 화면 — 공이 맞아 나갔나(넓은 경기장) 안 맞았나(선수 클로즈업). 해설 없이 영상으로.

중계는 공이 맞아 나가면 곧바로 넓은 경기장으로 카메라를 돌려 타구와 수비를 따라가고, 안 맞으면
(볼·스트라이크·삼진·볼넷) 타자·투수·포수 클로즈업을 보여 준다. 관절로 잰 사람 키(화면 높이
비율)로 가른다.

실측(2026-11-07, 1080p 하이라이트 한화-삼성 투구 45개, 기준값을 정한 재료): 투구 화면이 끝난 뒤
0.3~3.5초에 사람이 모두 작은(키 0.15 미만) 화면이 공이 맞아 나간 투구는 15~31장, 안 맞은
투구는 0장이었고 안 맞은 투구의 가장 큰 사람 키는 0.69~1.1이었다(10장/초).

실내 구장(고척돔, 처음 본 시험 경기)은 투구 뒤 클로즈업 배경이 어두운 벽·광고라 잔디·흙 색이
없어 관절 분석이 '경기장 화면 아님'으로 건너뛰었다 — 투구 21개 중 11개가 볼 프레임 0장으로
'모름'. 공이 맞아 나가면 카메라가 경기장을 따라가므로, 투구 뒤 3초 넘게 경기장 화면이 한 장도
없으면 안 맞은 것으로 본다. 조정용 3경기 투구 139개 중 이런 투구 6개는 정답이 삼진으로 보이는
아웃 5·볼넷 1이었고 공이 맞은 투구는 없었다.
"""

from collections.abc import Sequence
from typing import Optional

from app.domain.contact import MIN_SCORE, PeopleFrame

AFTER = (0.3, 3.5)  # 투구 화면이 끝난 뒤 볼 구간(초)
WIDE_TALL = 0.15  # 가장 큰 사람 키가 이보다 작으면 넓은 경기장 화면
CLOSE_TALL = 0.6  # 이보다 크면 선수 클로즈업
WIDE_SEC = 0.8  # 넓은 화면이 이만큼(초) 이어지면 공이 맞아 나갔다
MIN_JOINTS = 6
IN_PLAY, NO_PLAY = "in_play", "no_play"
VIEW_UNKNOWN = "view_unknown"  # 투구는 있었는데 화면으로 못 가름 — 앞 공의 화면을 대신 쓰지 않게


def tallest(frame: PeopleFrame) -> Optional[float]:
    """화면에서 가장 큰 사람의 키(관절 범위 세로 / 화면 높이). 사람이 없거나 높이를 모르면 None."""
    if not frame.height:
        return None
    best = None
    for person in frame.people:
        ys = [p[1] for p in person if p[2] >= MIN_SCORE]
        if len(ys) >= MIN_JOINTS:
            tall = (max(ys) - min(ys)) / frame.height
            best = tall if best is None else max(best, tall)
    return best


def play_view(frames: Sequence[PeopleFrame], t: float, fps: float = 10.0) -> Optional[str]:
    """투구 화면이 끝난 시각 `t` 뒤 — IN_PLAY · NO_PLAY · 모름(None).

    경기장이 아닌 화면(관중·그래픽·실내 클로즈업)은 관절 분석이 건너뛰어 프레임이 없다.
    구간 전체가 그랬고 분석이 구간 뒤까지 이어졌으면(영상·조각 끝이 아니면) 안 맞은 것으로 본다.
    """
    inside = [f for f in frames if t + AFTER[0] <= f.t <= t + AFTER[1]]
    if not inside and any(f.t > t + AFTER[1] for f in frames):
        return NO_PLAY
    seen = [m for f in inside for m in [tallest(f)] if m is not None]
    wide = sum(1 for m in seen if m < WIDE_TALL)
    if wide >= WIDE_SEC * fps:
        return IN_PLAY
    if seen and wide == 0 and max(seen) >= CLOSE_TALL:
        return NO_PLAY
    return None
