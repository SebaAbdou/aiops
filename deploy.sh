#!/bin/bash
# AIOps Kubernetes Deployment Script
# Builds Docker images and deploys all components to Kubernetes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
K8S_DIR="$SCRIPT_DIR/k8s"
REGISTRY="aiops"  # Local registry name (Docker Desktop)

echo "========================================"
echo "🚀 AIOps Kubernetes Deployment"
echo "========================================"

# Step 1: Build Docker Images
echo ""
echo "📦 Step 1: Building Docker images..."
echo ""

components=(
    "echo-server"
    "traffic-gen"
    "aiops-engine"
)

for component in "${components[@]}"; do
    echo "Building $component..."
    docker build -t "$REGISTRY/$component:latest" "$PROJECT_ROOT/$component"
    echo "✅ $component built successfully"
done

echo ""
echo "📦 All images built!"
docker images | grep "$REGISTRY"

# Step 2: Create namespace
echo ""
echo "🏗️  Step 2: Creating Kubernetes namespace..."
kubectl apply -f "$K8S_DIR/01-namespace.yaml"
echo "✅ Namespace created"

# Step 3: Deploy Prometheus
echo ""
echo "🔍 Step 3: Deploying Prometheus..."
kubectl apply -f "$K8S_DIR/02-prometheus-config.yaml"
kubectl apply -f "$K8S_DIR/03-pvc.yaml"
kubectl apply -f "$K8S_DIR/04-prometheus.yaml"
echo "✅ Prometheus deployed"
echo "⏳ Waiting for Prometheus to be ready..."
kubectl wait --for=condition=ready pod -l app=prometheus -n aiops --timeout=300s

# Step 4: Deploy Echo Server
echo ""
echo "🎯 Step 4: Deploying Echo Server..."
kubectl apply -f "$K8S_DIR/05-echo-server.yaml"
echo "✅ Echo Server deployed"
echo "⏳ Waiting for Echo Server to be ready..."
kubectl wait --for=condition=ready pod -l app=echo-server -n aiops --timeout=60s

# Step 5: Deploy Traffic Generator
echo ""
echo "🔄 Step 5: Deploying Traffic Generator..."
kubectl apply -f "$K8S_DIR/06-traffic-gen.yaml"
echo "✅ Traffic Generator deployed"
echo "⏳ Waiting for Traffic Generator to be ready..."
sleepfor=120
echo "Waiting $sleepfor seconds for Locust to start generating load..."
sleep $sleepfor

# Step 6: Deploy AIOps Engine
echo ""
echo "🧠 Step 6: Deploying AIOps Engine (Data Pipeline)..."
kubectl apply -f "$K8S_DIR/07-aiops-engine.yaml"
echo "✅ AIOps Engine deployed"

echo ""
echo "========================================"
echo "✨ Deployment Complete!"
echo "========================================"
echo ""
echo "📊 Cluster Status:"
kubectl get all -n aiops --no-headers

echo ""
echo "🔗 Service Endpoints:"
echo "  - Prometheus: http://localhost:9090"
echo "  - Echo Server: http://localhost:5000"
echo ""
echo "📈 Monitor logs:"
echo "  kubectl logs -f deployment/aiops-engine -n aiops"
echo ""
echo "💾 Get cleaned dataset:"
echo "  kubectl cp aiops/<pod-name>:/data/cleaned_dataset.csv ./cleaned_dataset.csv"
echo ""
echo "🗑️  Cleanup:"
echo "  kubectl delete namespace aiops"
