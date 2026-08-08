FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
COPY filing_room ./filing_room
RUN pip install --no-cache-dir uv==0.8.3 && uv sync --frozen --no-dev

USER 65532:65532
EXPOSE 8080
CMD ["/app/.venv/bin/uvicorn", "filing_room.main:app", "--host", "0.0.0.0", "--port", "8080"]
