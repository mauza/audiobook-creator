FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system -e .

RUN mkdir -p /app/input /app/output

# Copy source code last to optimize caching
COPY src/ src/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "audiobook_creator.cli"] 
