.PHONY: streamlit streamlit_fg db_up streamlit_docker docker_check k8s_up k8s_check k8s_dispatch

# Use Docker Desktop explicitly (avoids stale minikube docker-env in the shell).
DOCKER_CONTEXT = desktop-linux
COMPOSE = docker --context $(DOCKER_CONTEXT) compose -f orchestration/docker/docker-compose.yml --env-file .env

# New CMD window (detached from make); logs appear in that window, not here.
streamlit:
	cmd /c start "Streamlit" cmd /k "cd /d $(CURDIR) && make streamlit_fg

# Foreground in this terminal (logs here; use with venv already activated).
streamlit_fg:
	.venv\Scripts\activate.bat && make db_up && streamlit run streamlit_app.py

docker_check:
	@echo Checking Docker Desktop...
	cmd /c "docker --context $(DOCKER_CONTEXT) info >nul 2>&1 || (echo. & echo ERROR: Cannot reach Docker Desktop. & echo If you ran 'minikube docker-env' earlier, run: minikube docker-env --unset --shell powershell & echo Otherwise start Docker Desktop, wait until it is ready, then retry. & echo. & exit /b 1)"

db_up: docker_check
	@echo Starting Postgres (Compose)...
	$(COMPOSE) up -d

streamlit_docker:
	$(COMPOSE) --profile streamlit up -d

# Optional K8s setup. Requires Docker Desktop, minikube, and kubectl on PATH.
k8s_check:
	@echo Checking minikube and kubectl...
	cmd /c "where minikube >nul 2>&1 || (echo ERROR: minikube not found on PATH.& exit /b 1)"
	cmd /c "where kubectl >nul 2>&1 || (echo ERROR: kubectl not found on PATH.& exit /b 1)"

k8s_up: k8s_check db_up
	@echo Starting Minikube...
	minikube start
	@echo Applying K8s manifests...
	powershell -ExecutionPolicy Bypass -File orchestration/k8s/apply.ps1
	@echo === minikube status ===
	minikube status
	@echo === chess-teacher resources ===
	kubectl get cronjobs,pods,jobs -n chess-teacher
