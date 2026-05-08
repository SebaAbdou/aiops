@echo off
REM AIOps Kubernetes Deployment Script (Windows)

setlocal enabledelayedexpansion

set REGISTRY=aiops
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%
set K8S_DIR=%SCRIPT_DIR%k8s

echo.
echo ========================================
echo AIOps Kubernetes Deployment
echo ========================================
echo.

REM Step 1: Build Docker Images
echo.
echo Step 1: Building Docker images...
echo.

for %%C in (echo-server traffic-gen aiops-engine) do (
    echo Building %%C...
    docker build -t %REGISTRY%/%%C:latest "%PROJECT_ROOT%\%%C"
    if errorlevel 1 (
        echo ERROR: Failed to build %%C
        exit /b 1
    )
    echo OK: %%C built successfully
)

echo.
echo All images built!
docker images | findstr %REGISTRY%

REM Step 2: Create namespace
echo.
echo Step 2: Creating Kubernetes namespace...
kubectl apply -f "%K8S_DIR%\01-namespace.yaml"
if errorlevel 1 (
    echo ERROR: Failed to create namespace
    exit /b 1
)
echo OK: Namespace created

REM Step 3: Deploy Prometheus
echo.
echo Step 3: Deploying Prometheus...
kubectl apply -f "%K8S_DIR%\02-prometheus-config.yaml"
kubectl apply -f "%K8S_DIR%\03-pvc.yaml"
kubectl apply -f "%K8S_DIR%\04-prometheus.yaml"
if errorlevel 1 (
    echo ERROR: Failed to deploy Prometheus
    exit /b 1
)
echo OK: Prometheus deployed
echo Waiting for Prometheus to be ready...
kubectl wait --for=condition=ready pod -l app=prometheus -n aiops --timeout=300s

REM Step 4: Deploy Echo Server
echo.
echo Step 4: Deploying Echo Server...
kubectl apply -f "%K8S_DIR%\05-echo-server.yaml"
if errorlevel 1 (
    echo ERROR: Failed to deploy Echo Server
    exit /b 1
)
echo OK: Echo Server deployed
echo Waiting for Echo Server to be ready...
kubectl wait --for=condition=ready pod -l app=echo-server -n aiops --timeout=60s

REM Step 5: Deploy Traffic Generator
echo.
echo Step 5: Deploying Traffic Generator...
kubectl apply -f "%K8S_DIR%\06-traffic-gen.yaml"
if errorlevel 1 (
    echo ERROR: Failed to deploy Traffic Generator
    exit /b 1
)
echo OK: Traffic Generator deployed
echo Waiting 120 seconds for Locust to generate metrics...
timeout /t 120 /nobreak

REM Step 6: Deploy AIOps Engine
echo.
echo Step 6: Deploying AIOps Engine...
kubectl apply -f "%K8S_DIR%\07-aiops-engine.yaml"
if errorlevel 1 (
    echo ERROR: Failed to deploy AIOps Engine
    exit /b 1
)
echo OK: AIOps Engine deployed

echo.
echo ========================================
echo Deployment Complete!
echo ========================================
echo.

echo Cluster Status:
kubectl get all -n aiops --no-headers

echo.
echo Service Endpoints:
echo   - Prometheus: http://localhost:9090
echo   - Echo Server: http://localhost:5000
echo.
echo Monitor logs:
echo   kubectl logs -f deployment/aiops-engine -n aiops
echo.
echo Get cleaned dataset:
echo   kubectl cp aiops/^<pod-name^>:/data/cleaned_dataset.csv ./cleaned_dataset.csv
echo.
echo Cleanup:
echo   kubectl delete namespace aiops
