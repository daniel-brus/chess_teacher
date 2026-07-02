.PHONY: streamlit streamlit_fg streamlit_docker docker_check k8s_up k8s_check k8s_ensure streamlit_k8s

# Use Docker Desktop explicitly (stable context for Compose and k3d).
DOCKER_CONTEXT = desktop-linux
K3D_CLUSTER = chess-teacher
COMPOSE = docker --context $(DOCKER_CONTEXT) compose --env-file .env

# APP_PORT in .env: host port for local venv + Compose streamlit. k3d stays on 8501.
# New CMD window (detached from make); logs appear in that window, not here.
streamlit:
	cmd /c start "Streamlit" cmd /k "cd /d $(CURDIR) && make streamlit_fg"

# Foreground in this terminal (logs here; use with venv already activated).
streamlit_fg:
	.venv\Scripts\activate.bat && python _load_dotenv.py

docker_check:
	@echo Checking Docker Desktop...
	cmd /c "docker --context $(DOCKER_CONTEXT) info >nul 2>&1 || (echo. & echo ERROR: Cannot reach Docker Desktop. & echo Start Docker Desktop, wait until it is ready, then retry. & echo. & exit /b 1)"

streamlit_docker:
	$(COMPOSE) up -d

# Optional K8s setup. Requires Docker Desktop, k3d, and kubectl on PATH.
k8s_check:
	@cmd /c "where k3d >nul 2>&1 && where kubectl >nul 2>&1 || (echo k3d and kubectl must be on PATH & exit /b 1)"

k8s_ensure: k8s_check
	powershell -ExecutionPolicy Bypass -File orchestration/k8s/ensure-cluster.ps1

k8s_up: k8s_ensure
	powershell -ExecutionPolicy Bypass -File orchestration/k8s/apply.ps1

# Port-forward Streamlit for localhost + Caddy (host.docker.internal:8501).
# --address 0.0.0.0 required: default bind is 127.0.0.1 only, Docker cannot reach that.
# ONLY DESIGNED TO USE UNTIL STREAMLIT IS RUNNING ON VPS
streamlit_k8s: k8s_check
	@cmd /c "start /B kubectl port-forward --address 0.0.0.0 -n chess-teacher svc/streamlit 8501:8501 1>nul 2>nul"
