#!/bin/bash
# Install dependencies in Triton container

CONTAINER_NAME="triton-ocr-ensemble"

echo "Installing Python dependencies in Triton container..."

podman exec $CONTAINER_NAME pip install --no-cache-dir shapely pyclipper opencv-python-headless

echo "✓ Dependencies installed"
