FROM python:3.12-slim

# Installeer Stockfish
RUN apt-get update && apt-get install -y stockfish && rm -rf /var/lib/apt/lists/*

# Werkdirectory
WORKDIR /app

# Dependencies eerst (cache-vriendelijk)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Config en source code
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/

# Data map aanmaken
RUN mkdir -p storage

CMD ["python", "-m", "scripts.main"]
