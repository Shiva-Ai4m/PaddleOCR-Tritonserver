#!/bin/bash

# Run Triton Inference Server with OCR Pipeline

TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.08-py3"
CONTAINER_NAME="triton-ocr"
MODEL_REPO="$(pwd)/triton_models_ensemble"

echo "Starting Triton Inference Server (OCR Pipeline)..."
echo "Image: $TRITON_IMAGE"
echo "Model Repository: $MODEL_REPO"

# Stop and remove existing container if it exists
podman stop $CONTAINER_NAME 2>/dev/null
podman rm $CONTAINER_NAME 2>/dev/null

# Run Triton server
podman run -d \
  --name $CONTAINER_NAME \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $MODEL_REPO:/models \
  --shm-size=2g \
  $TRITON_IMAGE \
  bash -c "pip install -q opencv-python-headless shapely pyclipper && \
           tritonserver --model-repository=/models"

echo ""
echo "Triton server is starting..."
echo "Waiting for server to be ready..."
sleep 30

# Check if container is running
if podman ps | grep -q $CONTAINER_NAME; then
    echo "✓ Container is running"
    echo ""

    # Wait for models to load
    echo "Waiting for models to load..."
    for i in {1..20}; do
        if curl -s http://localhost:8000/v2/health/ready 2>&1 | grep -q "true"; then
            echo "✓ Server is ready!"
            echo ""
            break
        fi
        echo "  Waiting... ($i/20)"
        sleep 2
    done

    echo ""
    echo "✓ Triton OCR Server is ready!"
    echo ""
    echo "Endpoints:"
    echo "  HTTP:    http://localhost:8000"
    echo "  gRPC:    localhost:8001"
    echo "  Metrics: http://localhost:8002"
    echo ""
    echo "Usage:"
    echo "  python3 ocr_pipeline_client.py <image.jpg>"
    echo "  python3 ocr_pipeline_client.py <image.jpg> --visualize --output result.json"
    echo ""
    echo "Commands:"
    echo "  View logs:   podman logs -f $CONTAINER_NAME"
    echo "  Stop server: podman stop $CONTAINER_NAME"
else
    echo "✗ Container failed to start"
    echo "Check logs: podman logs $CONTAINER_NAME"
    exit 1
fi
