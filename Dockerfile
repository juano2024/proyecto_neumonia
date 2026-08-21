FROM python:3.10-slim

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3-opencv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

CMD ["uv", "run", "pytest", "-v"]