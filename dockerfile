FROM python:3.12-slim

# Installeer Stockfish
RUN apt-get update && apt-get install -y stockfish && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install source code as package
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install .

# Copy scripts and app-files
COPY scripts/ ./scripts/
COPY streamlit_app.py .
COPY streamlit_utils/ ./streamlit_utils/
COPY pages/ ./pages/

RUN mkdir -p storage

CMD ["streamlit","run","streamlit_app.py","--server.address","0.0.0.0","--server.port","8501"]
