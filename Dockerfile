FROM nvidia/cuda:13.1.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages uv

WORKDIR /app

# Copy source and project metadata
COPY pyproject.toml .
COPY src/ src/

RUN uv pip install --system -e .

RUN mkdir -p /app/input /app/output

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python3", "-m", "audiobook_creator.cli"]
