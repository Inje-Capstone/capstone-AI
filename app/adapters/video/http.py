"""영상 분석 백엔드 공통 HTTP. 의존성 추가 없이 urllib만 쓴다."""

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from app.adapters.video.base import VideoAnalyzerError

# 재시도는 5xx·네트워크 오류만. 4xx·429는 재시도하면 쿼터만 두 배로 탄다(RELIABILITY).
_RETRYABLE = frozenset({500, 502, 503, 504})


def post_json(
    url: str,
    payload: dict[str, Any],
    token: Optional[str] = None,
    timeout: float = 300.0,
    retries: int = 2,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last: Optional[Exception] = None
    for _ in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:  # noqa: S310
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code not in _RETRYABLE:
                raise VideoAnalyzerError(f"HTTP {exc.code}: {detail}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
    raise VideoAnalyzerError(f"요청 실패: {url} ({last})")


def completion_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VideoAnalyzerError(
            f"choices[0].message.content가 없다: {str(response)[:200]}"
        ) from exc
    if isinstance(content, list):  # 일부 서버는 content 파트 배열로 준다
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)
