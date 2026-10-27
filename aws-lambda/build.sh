#!/usr/bin/env bash
set -euo pipefail

HOST_UID=$(id -u)
HOST_GID=$(id -g)

echo "Cleaning previous build..."
sudo rm -rf package lambda-deployment.zip
mkdir -p package

echo "Installing Lambda dependencies in Docker..."
docker run --rm \
  --platform linux/amd64 \
  --entrypoint /bin/bash \
  -e HOST_UID="$HOST_UID" \
  -e HOST_GID="$HOST_GID" \
  -v "$PWD:/var/task" \
  -w /var/task \
  python:3.12-slim \
  -c 'python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt -t package && chown -R "$HOST_UID:$HOST_GID" package'

echo "Removing unnecessary files..."
find package -type d \( \
  -name tests -o \
  -name __pycache__ -o \
  -name '*.dist-info' \
\) -prune -exec rm -rf {} +

find package -type f \( \
  -name '*.pyc' -o \
  -name '*.pyo' \
\) -delete

echo "Copying Lambda code and model..."
cp lambda_function.py package/
mkdir -p package/model
cp model/road_risk_model.joblib package/model/

echo "Creating deployment ZIP..."
(
  cd package
  zip -qr ../lambda-deployment.zip .
)

echo "Build completed."
ls -lh lambda-deployment.zip
du -sh package
