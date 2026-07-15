.PHONY: streamlit streamlit_fg streamlit_docker streamlit_secrets docker_check doppler_check \
	dev_infra dev_down dev_bootstrap dev_sync_cloud dev_bootstrap_schema \
	k8s_up k8s_check k8s_ensure streamlit_k8s

# Use Docker Desktop explicitly (stable context for Compose and k3d).
DOCKER_CONTEXT = desktop-linux
K3D_CLUSTER = chess-teacher
COMPOSE = docker --context $(DOCKER_CONTEXT) compose
COMPOSE_INFRA = $(COMPOSE) -f docker-compose.infra.yml
COMPOSE_ALL = $(COMPOSE) -f docker-compose.infra.yml -f docker-compose.yml

# Doppler project/config names (no doppler.yaml in repo — flags are explicit).
DOPPLER_PROJECT = chess-teacher
DOPPLER_CONFIG_LOCAL = dev_local
DOPPLER_CONFIG_K3D = dev_k3d
DOPPLER_RUN_LOCAL = doppler run --project $(DOPPLER_PROJECT) --config $(DOPPLER_CONFIG_LOCAL) --
DOPPLER_RUN_K3D = doppler run --project $(DOPPLER_PROJECT) --config $(DOPPLER_CONFIG_K3D) --

# Interim: make streamlit_fg DOPPLER_CONFIG_LOCAL=prod  (cloud backends until dev_local is filled)
streamlit: doppler_check
	cmd /c start "Streamlit" cmd /k "cd /d $(CURDIR) && make streamlit_fg"

doppler_check:
	@cmd /c "where doppler >nul 2>&1 || (echo. & echo ERROR: Doppler CLI not found. & echo Install: winget install doppler.doppler & echo Then: doppler login & echo. & exit /b 1)"

streamlit_secrets: doppler_check
	$(DOPPLER_RUN_LOCAL) .venv\Scripts\python.exe scripts/dev/render_streamlit_secrets.py

streamlit_fg: doppler_check streamlit_secrets
	$(DOPPLER_RUN_LOCAL) .venv\Scripts\python.exe scripts/run_streamlit.py

docker_check:
	@echo Checking Docker Desktop...
	cmd /c "docker --context $(DOCKER_CONTEXT) info >nul 2>&1 || (echo. & echo ERROR: Cannot reach Docker Desktop. & echo Start Docker Desktop, wait until it is ready, then retry. & echo. & exit /b 1)"

dev_infra: docker_check doppler_check
	$(DOPPLER_RUN_LOCAL) $(COMPOSE_INFRA) up -d

dev_down: docker_check
	-$(COMPOSE_ALL) down

dev_bootstrap: docker_check doppler_check
	powershell -ExecutionPolicy Bypass -File scripts/dev/bootstrap_local.ps1

streamlit_docker: docker_check doppler_check streamlit_secrets dev_infra
	$(DOPPLER_RUN_LOCAL) $(COMPOSE_ALL) up -d --build

dev_sync_cloud: doppler_check docker_check dev_infra
	powershell -ExecutionPolicy Bypass -File scripts/dev/sync_from_cloud.ps1

dev_bootstrap_schema: doppler_check
	$(DOPPLER_RUN_LOCAL) .venv\Scripts\python.exe scripts/dev/bootstrap_schema.py

k8s_check:
	@cmd /c "where k3d >nul 2>&1 && where kubectl >nul 2>&1 || (echo k3d and kubectl must be on PATH & exit /b 1)"

k8s_ensure: k8s_check
	powershell -ExecutionPolicy Bypass -File orchestration/k8s/ensure-cluster.ps1

k8s_up: k8s_ensure doppler_check dev_infra
	$(DOPPLER_RUN_K3D) powershell -ExecutionPolicy Bypass -File orchestration/k8s/apply.ps1

streamlit_k8s: k8s_check
	@cmd /c "start /B kubectl port-forward --address 0.0.0.0 -n chess-teacher svc/streamlit 8501:8501 1>nul 2>nul"
