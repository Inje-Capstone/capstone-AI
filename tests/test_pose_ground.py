"""몸 아래 색 — 잔디(외야 다이빙)와 흙(주자 슬라이딩)."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from app.adapters.video.pose import ground_under  # noqa: E402

GRASS = (40, 150, 40)  # BGR
DIRT = (60, 110, 190)


def picture(below):
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (90, 90, 90)
    img[120:200, :] = below  # 몸 아래
    return img


def test_body_lying_on_grass():
    grass, dirt = ground_under(picture(GRASS), (50, 80, 150, 120))
    assert grass > 0.9 and dirt < 0.1


def test_body_on_dirt():
    grass, dirt = ground_under(picture(DIRT), (50, 80, 150, 120))
    assert dirt > 0.9 and grass < 0.1


def test_box_at_the_bottom_edge_gives_zero_not_an_error():
    assert ground_under(picture(GRASS), (50, 180, 150, 200)) == (0.0, 0.0)
