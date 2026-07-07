FROM python:3.12-slim

# Install Stockfish (Debian package installs to /usr/games/stockfish).
RUN apt-get update && apt-get install -y stockfish && rm -rf /var/lib/apt/lists/*
ENV STOCKFISH_PATH=/usr/games/stockfish

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install source code as package
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install .

# Copy scripts and app-files# Copy scripts and app-files (including k8s/job/ for dispatcher)
COPY scripts/ ./scripts/
COPY orchestration/k8s/job/ ./orchestration/k8s/job/
COPY streamlit_app.py .
COPY streamlit_utils/ ./streamlit_utils/
COPY streamlit_pages/ ./streamlit_pages/
COPY .streamlit/ ./.streamlit/

RUN mkdir -p storage

CMD ["streamlit","run","streamlit_app.py","--server.address","0.0.0.0","--server.port","8501"]
