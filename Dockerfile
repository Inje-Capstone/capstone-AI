FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 의존성 레이어를 먼저 굳혀 코드만 바뀔 때 재설치를 피한다.
# 패키지 자체는 설치하지 않는다 — /app에서 import해야 config.ROOT가 data/를 가리킨다.
COPY pyproject.toml ./
RUN python -c "import tomllib; print(*tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies'], sep=chr(10))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts

# 키는 이미지에 굽지 않는다 — 실행 시 환경변수(ANTHROPIC_API_KEY)로 주입.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
