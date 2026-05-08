#!/bin/bash
# MLOps Automation Stack Deployment Script
# 
# Deploys the complete automated ML pipeline to Kubernetes:
# - CronJob for continuous drift detection (every 5 min)
# - Job template for automated retraining
# - Grafana dashboard for visualization
# - ConfigMaps and PVCs for status tracking
#
# Usage: ./deploy-mlops-stack.sh
# Prerequisites: kubectl configured, aiops namespace exists

set -e

echo "=========================================="
echo "AIOps MLOps Automation Stack Deployment"
echo "=========================================="

# Configuration
NAMESPACE="aiops"
K8S_DIR="./k8s"
IMAGE_REGISTRY="docker.io"  # Adjust to your registry
IMAGE_NAME="aiops-engine:latest"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}ℹ ${1}${NC}"
}

log_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠ ${1}${NC}"
}

log_error() {
    echo -e "${RED}✗ ${1}${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_error "Namespace $NAMESPACE not found. Create it first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Deploy ConfigMaps and PVCs
deploy_config_and_pvc() {
    log_info "Deploying ConfigMaps and PVCs..."
    
    kubectl apply -f "$K8S_DIR/16-mlops-config.yaml"
    
    if kubectl get pvc -n $NAMESPACE aiops-shared &> /dev/null; then
        log_success "Shared PVC already exists"
    else
        log_error "Shared PVC creation failed"
        exit 1
    fi
}

# Deploy Grafana Dashboard
deploy_grafana_dashboard() {
    log_info "Deploying Grafana dashboard ConfigMap..."
    
    kubectl apply -f "$K8S_DIR/15-grafana-mlops-dashboard.yaml"
    log_success "Grafana dashboard ConfigMap deployed"
    
    log_warn "NOTE: Import the dashboard in Grafana:"
    log_warn "  1. Go to Grafana: Configuration > Data Sources > Prometheus"
    log_warn "  2. Create/verify Prometheus datasource pointing to prometheus:9090"
    log_warn "  3. Import dashboard from ConfigMap: aiops_mlops"
}

# Deploy Drift Detection CronJob
deploy_drift_detection() {
    log_info "Deploying drift detection CronJob..."
    
    kubectl apply -f "$K8S_DIR/13-drift-detection-cronjob.yaml"
    
    if kubectl get cronjob -n $NAMESPACE aiops-drift-detection &> /dev/null; then
        log_success "Drift detection CronJob deployed"
    else
        log_error "CronJob deployment failed"
        exit 1
    fi
}

# Deploy Retraining Job Template
deploy_retraining_job() {
    log_info "Deploying retraining Job template..."
    
    kubectl apply -f "$K8S_DIR/14-retraining-job.yaml"
    log_success "Retraining Job template deployed"
}

# Verify RBAC
verify_rbac() {
    log_info "Verifying RBAC permissions..."
    
    if kubectl get sa -n $NAMESPACE aiops-worker &> /dev/null; then
        log_success "ServiceAccount aiops-worker exists"
    else
        log_error "ServiceAccount not found"
        exit 1
    fi
    
    if kubectl get clusterrole aiops-worker &> /dev/null; then
        log_success "ClusterRole aiops-worker exists"
    else
        log_error "ClusterRole not found"
        exit 1
    fi
}

# Check image availability
check_image() {
    log_info "Checking Docker image availability..."
    
    if docker pull $IMAGE_NAME &> /dev/null; then
        log_success "Image $IMAGE_NAME is available"
    else
        log_warn "Image $IMAGE_NAME not found locally"
        log_warn "Make sure to build and push it before CronJob runs"
    fi
}

# Display status
display_status() {
    log_info "Displaying deployment status..."
    
    echo ""
    echo "CronJobs:"
    kubectl get cronjob -n $NAMESPACE -l app=aiops
    
    echo ""
    echo "ServiceAccounts:"
    kubectl get sa -n $NAMESPACE -l app=aiops
    
    echo ""
    echo "ConfigMaps:"
    kubectl get cm -n $NAMESPACE | grep mlops
    
    echo ""
    echo "PVC:"
    kubectl get pvc -n $NAMESPACE | grep aiops
}

# Test drift detection
test_drift_detection() {
    log_info "Running manual test of drift detection..."
    
    # Create a test job
    JOB_NAME="aiops-drift-detection-test-$(date +%s)"
    
    kubectl create job $JOB_NAME --from=cronjob/aiops-drift-detection -n $NAMESPACE
    
    log_info "Test job created: $JOB_NAME"
    log_info "Check pod logs with: kubectl logs -f -n $NAMESPACE -l job-name=$JOB_NAME"
    log_warn "Job may take a few minutes to complete"
}

# Main deployment flow
main() {
    log_info "Starting MLOps stack deployment..."
    
    # Run deployment steps
    check_prerequisites
    deploy_config_and_pvc
    deploy_grafana_dashboard
    deploy_drift_detection
    deploy_retraining_job
    verify_rbac
    check_image
    
    echo ""
    display_status
    
    echo ""
    log_success "MLOps automation stack deployed successfully!"
    
    echo ""
    echo "Next steps:"
    echo "1. Build and push the aiops-engine Docker image:"
    echo "   docker build -t $IMAGE_NAME aiops-engine/"
    echo "   docker push $IMAGE_NAME"
    echo ""
    echo "2. Verify the drift detection CronJob is running:"
    echo "   kubectl get cronjob -n $NAMESPACE"
    echo "   kubectl get pods -n $NAMESPACE -l app=aiops"
    echo ""
    echo "3. View real-time status in Grafana:"
    echo "   kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000"
    echo "   Open http://localhost:3000 and import mlops dashboard"
    echo ""
    echo "4. Monitor logs:"
    echo "   kubectl logs -f -n $NAMESPACE -l app=aiops,job=drift-detection"
    echo ""
    echo "5. (Optional) Run manual drift detection test:"
    read -p "Run test now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_drift_detection
    fi
}

# Run main function
main
