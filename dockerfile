FROM python:3.12-slim

# Installeer Stockfish
RUN apt-get update && apt-get install -y stockfish && rm -rf /var/lib/apt/lists/*

# Werkdirectory
WORKDIR /app

# Dependencies eerst (cache-vriendelijk)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app.py ./

# Data map aanmaken
RUN mkdir -p storage

# NOG VERANDEREN
CMD ["streamlit","run","app.py","--server.address","0.0.0.0","--server.port","8501"]
