#!/bin/bash
#
# Build Lambda deployment zips inside Docker (Python 3.11, Linux x86_64)
# so binaries match AWS Lambda. Use this if local build gives ImportModuleError
# (e.g. numpy built for wrong Python version on Mac/Windows).
#
# Requires: Docker. Run from project root.
#   ./scripts/build_lambda_packages_docker.sh
#
set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

check_docker() {
    if ! command -v docker &>/dev/null; then
        echo "Docker is required. Install Docker Desktop or Docker Engine."
        exit 1
    fi
}

# Use AWS Lambda Python 3.11 base image so glibc and libs match Lambda (avoids GLIBC_2.27 not found)
IMAGE="public.ecr.aws/lambda/python:3.11"

build_xgb() {
    echo "Building XGB-only package in Docker (numpy, xgboost, scipy)..."
    docker run --rm --entrypoint bash \
        -v "${PROJECT_ROOT}:/workspace" \
        -w /workspace \
        "$IMAGE" \
        -c '
            rm -rf _docker_build_xgb
            /var/lang/bin/pip install --no-cache-dir --only-binary :all: "numpy<2" xgboost scipy -t _docker_build_xgb --no-deps --quiet
            find _docker_build_xgb -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            find _docker_build_xgb -name "*.pyc" -delete 2>/dev/null || true
        '
    cd "$PROJECT_ROOT"
    cp lambda/handler.py _docker_build_xgb/
    mkdir -p _docker_build_xgb/models
    cp models/adult_xgb.json models/cancer_xgb.json _docker_build_xgb/models/ 2>/dev/null || true
    rm -f lambda_xgb.zip
    (cd _docker_build_xgb && zip -rq ../lambda_xgb.zip .)
    rm -rf _docker_build_xgb
    echo "  -> lambda_xgb.zip ($(du -h lambda_xgb.zip | cut -f1))"
}

build_mlp() {
    echo "Building MLP-only package in Docker (numpy, onnxruntime)..."
    docker run --rm --entrypoint bash \
        -v "${PROJECT_ROOT}:/workspace" \
        -w /workspace \
        "$IMAGE" \
        -c '
            rm -rf _docker_build_mlp
            /var/lang/bin/pip install --no-cache-dir --only-binary :all: "numpy<2" onnxruntime -t _docker_build_mlp --no-deps --quiet
            find _docker_build_mlp -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            find _docker_build_mlp -name "*.pyc" -delete 2>/dev/null || true
        '
    cd "$PROJECT_ROOT"
    cp lambda/handler.py _docker_build_mlp/
    mkdir -p _docker_build_mlp/models
    cp models/adult_mlp.onnx models/cancer_mlp.onnx _docker_build_mlp/models/ 2>/dev/null || true
    rm -f lambda_mlp.zip
    (cd _docker_build_mlp && zip -rq ../lambda_mlp.zip .)
    rm -rf _docker_build_mlp
    echo "  -> lambda_mlp.zip ($(du -h lambda_mlp.zip | cut -f1))"
}

check_docker
build_xgb
build_mlp
echo "Done. Upload lambda_xgb.zip or lambda_mlp.zip to Lambda (Python 3.11)."
