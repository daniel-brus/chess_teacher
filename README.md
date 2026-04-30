# Chess Teacher

An AI-powered chess teaching application with a Streamlit UI for visualizing and analyzing chess games. Built with Python, Docker, and the Anthropic API.

## Features

- 🎯 Interactive chess board visualization with Streamlit
- 🤖 AI-powered chess analysis via Anthropic API
- 📊 Move history and game statistics
- 💾 Persistent game storage with SQLite
- ⚙️ Stockfish integration for move suggestions
- 🧪 Comprehensive test suite with pytest
- 🔧 Pre-commit hooks for code quality

## Setup

### Prerequisites
- Python 3.12
- Docker & Docker Compose (optional)
- Stockfish (included in Docker, or install locally)

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/daniel-brus/chess_teacher.git
   cd chess_teacher
   ```

2. **Create and activate virtual environment:**
   ```bash
   py -3.12 -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Configure environment:**
   - Copy `.env.example` to `.env`
   - Add your API keys (DOCKERHUB_USERNAME, DOCKERHUB_TOKEN, etc.)
   - Update `config.env` if needed

5. **Run tests:**
   ```bash
   pytest
   ```

6. **Start Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`

7. **Or run backend scripts:**

Example:
   ```bash
   python scripts/main.py
   ```

### Docker

```bash
docker-compose up
```

## Project Structure

```CLI entry points
app.py            # Streamlit web interface
src/
  database/       # Database operations
  utils/          # Utility functions
config/           # Configuration management
tests/            # Test suite (mirrors src/)
scripts/          # Entry points
```

## Development Tools
- **Web UI:** Streamlit for interactive chess visualization
- **Linting & Formatting:** Ruff (configured in `pyproject.toml`)
- **Pre-commit Hooks:** Automatic checks before commits
- **Testing:** Pytest with fixtures in `tests/conftest.py`
- **Chess Engine:** Stockfish for move analysis

### Code Quality

Use VS Code with Ruff extension enabled (auto-format on save).

Otherwise, do the following before committing:
```bash
ruff check src config tests --fix
pre-commit run --all-files
```
