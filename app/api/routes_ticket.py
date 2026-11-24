"""티켓 인증(백엔드 전용) — 사진을 읽어 경기 일자·구장·좌석을 돌려주고 말이 되는지 검사한다.

`/api/backend/**`는 백엔드 중계에서 막혀 있어 프론트가 직접 부르지 않는다. 백엔드가 사진을 받아
여기로 넘기고, `seatKey` 중복·주 2회·40점은 백엔드가 판단한다. 사진은 저장·로그하지 않는다.
"""

import base64
import binascii
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.adapters.vision.ticket import TicketReaderError
from app.api.deps import get_ticket_reader
from app.api.schemas import (
    BackendTicketCheckOut,
    BackendTicketReadIn,
    BackendTicketReadOut,
    BackendTicketSeatOut,
)
from app.domain.ticket import check_ticket

router = APIRouter(prefix="/api/backend/tickets", tags=["backend"])

KST = timezone(timedelta(hours=9))
# Claude API 이미지 한도(5MB)는 base64 길이 기준이라 원본은 그 3/4까지. 크면 백엔드가 줄여 보낸다.
MAX_BASE64_CHARS = 5 * 1024 * 1024
MAX_IMAGE_BYTES = MAX_BASE64_CHARS * 3 // 4
SNIFF = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_media_type(image: bytes) -> Optional[str]:
    for magic, media in SNIFF:
        if image.startswith(magic):
            return media
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    return None


def today_kst():
    return datetime.now(KST).date()


@router.post("/read", response_model=BackendTicketReadOut, summary="티켓 사진 판독·검사")
def read_ticket(body: BackendTicketReadIn):
    if len(body.imageBase64) > MAX_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="사진이 너무 큽니다(원본 3.75MB 이하)")
    try:
        image = base64.b64decode(body.imageBase64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="imageBase64가 올바른 base64가 아닙니다"
        ) from exc
    if len(image) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="사진이 너무 큽니다(원본 3.75MB 이하)")
    media = sniff_media_type(image)
    if media is None:
        raise HTTPException(status_code=415, detail="jpeg·png·webp·gif 사진만 받습니다")

    reader = get_ticket_reader()
    try:
        raw = reader.read(image, media)  # 선언된 mediaType이 틀려도 바이트로 판단한 형식을 쓴다
    except TicketReaderError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    c = check_ticket(raw, today_kst())
    return BackendTicketReadOut(
        ok=c.ok,
        reasons=[x.detail for x in c.checks if not x.ok],
        gameDate=c.game_date, stadium=c.stadium, stadiumText=c.stadium_text,
        homeTeam=c.home_team, awayTeam=c.away_team,
        seat=BackendTicketSeatOut(section=c.section, block=c.block, row=c.row, number=c.number),
        kind=c.kind, seatKey=c.seat_key,
        checks=[BackendTicketCheckOut(name=x.name, ok=x.ok, detail=x.detail) for x in c.checks],
        scheduleChecked=False,
        model=reader.model,
    )
