.PHONY: streamlit streamlit_fg db_up streamlit_docker

# New CMD window (detached from make); logs appear in that window, not here.
streamlit:
	cmd /c start "Streamlit" cmd /k "cd /d $(CURDIR) && make streamlit_fg

# Foreground in this terminal (logs here; use with venv already activated).
streamlit_fg:
	.venv\Scripts\activate.bat && make db_up && streamlit run streamlit_app.py

db_up:
	docker compose -f orchestration/docker/docker-compose.yml --env-file .env up -d

streamlit_docker:
	docker compose -f orchestration/docker/docker-compose.yml --profile streamlit --env-file .env up -d
