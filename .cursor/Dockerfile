FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System toolchain + local dev backends (Postgres, Redis) + Stockfish engine.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      build-essential libpq-dev pkg-config \
      postgresql postgresql-contrib \
      redis-server \
      stockfish \
      curl ca-certificates git sudo \
    && rm -rf /var/lib/apt/lists/*

# MinIO server + client (S3-compatible object storage for local dev).
RUN curl -fsSL https://dl.min.io/server/minio/release/linux-amd64/minio -o /usr/local/bin/minio \
    && curl -fsSL https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc \
    && chmod +x /usr/local/bin/minio /usr/local/bin/mc

# The base ubuntu:24.04 image ships an 'ubuntu' user (uid 1000); give it passwordless sudo
# so the agent can manage the Postgres cluster and other services.
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

ENV STOCKFISH_PATH=/usr/games/stockfish

USER ubuntu
