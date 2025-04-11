#!/bin/bash

# Kubernetes deployment script for LightMultiDetect

# Check for required tools
command -v kubectl >/dev/null 2>&1 || { echo >&2 "kubectl is required but not installed. Aborting."; exit 1; }

# Set namespace
NAMESPACE="lightmultidetect"

# Create namespace if it doesn't exist
kubectl get namespace $NAMESPACE > /dev/null 2>&1 || kubectl create namespace $NAMESPACE

# Apply configurations
echo "Applying Kubernetes configurations..."

# Apply configs
kubectl apply -f kubernetes/config.yaml -n $NAMESPACE
kubectl apply -f kubernetes/secrets.yaml -n $NAMESPACE

# Apply persistent volumes
kubectl apply -f kubernetes/persistent-volumes.yaml -n $NAMESPACE

# Apply deployments
kubectl apply -f kubernetes/deployments.yaml -n $NAMESPACE

# Apply services
kubectl apply -f kubernetes/services.yaml -n $NAMESPACE

# Apply ingress
kubectl apply -f kubernetes/ingress.yaml -n $NAMESPACE

# Apply network policies
kubectl apply -f kubernetes/network-policy.yaml -n $NAMESPACE

# Apply autoscaling
kubectl apply -f kubernetes/horizontal-pod-autoscaler.yaml -n $NAMESPACE

# Check deployment status
echo "Checking deployment status..."
kubectl get pods -n $NAMESPACE

echo "Kubernetes deployment completed!"
echo "The application will be available at: https://lightmultidetect.example.com (once DNS is configured)" 