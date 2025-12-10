# Setup Guide - PaddleOCR Triton Inference Server

Complete guide to set up and run the OCR pipeline from scratch.

## System Requirements

- **OS**: Linux (tested on RHEL 9.5)
- **Python**: 3.9+
- **Container Runtime**: Docker or Podman
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: ~2GB for models and dependencies

## Quick Start

```bash
# 1. Clone repository
git clone <your-repo-url>
cd paddleocr-triton

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Triton server
./run_triton_ensemble.sh

# 4. Run OCR (in another terminal)
python3 test_ocr_final_correct.py input.jpg output.jpg
```

## Detailed Setup

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `tritonclient[grpc]` - Triton client library
- `opencv-python-headless` - Image processing
- `numpy` - Numerical operations
- `shapely` - Polygon operations for detection
- `pyclipper` - Box expansion for text detection
- `Pillow` - Image I/O

### Step 2: Verify Model Files

```bash
ls -lh triton_models_ensemble/ocr_detection/1/*.onnx
ls -lh triton_models_ensemble/ocr_recognition/1/*.onnx
```

Expected files:
- `en_PP-OCRv3_det.onnx` (~2.3 MB)
- `en_PP-OCRv3_rec.onnx` (~8.5 MB)

### Step 3: Start Triton Server

```bash
./run_triton_ensemble.sh
```

Wait ~30 seconds for models to load.

### Step 4: Verify Server

```bash
python3 -c "import tritonclient.grpc as grpcclient; \
client = grpcclient.InferenceServerClient('localhost:8001'); \
print('✓ Server ready:', client.is_server_ready())"
```

### Step 5: Run OCR

```bash
python3 test_ocr_final_correct.py <image.jpg> [output.jpg]
```

## Troubleshooting

### Missing Dependencies

```bash
pip install tritonclient[grpc] opencv-python-headless numpy shapely pyclipper Pillow
```

### Character Dictionary Not Found

The script needs `../ppocr/utils/en_dict.txt`. Either:
1. Clone full PaddleOCR repo as parent
2. Modify path in `test_ocr_final_correct.py` line 14

### Server Connection Failed

```bash
# Check container
podman ps | grep triton

# View logs
podman logs triton-ocr

# Restart server
podman stop triton-ocr
./run_triton_ensemble.sh
```

## Configuration

### Git Setup

```bash
git config user.name "Shivashankarar"
git config user.email "shivashankarar@ai4mtech.com"
```

### Port Configuration

Modify `run_triton_ensemble.sh` if ports are in use:
```bash
-p 9000:8000  # HTTP
-p 9001:8001  # gRPC
-p 9002:8002  # Metrics
```

## Performance Tips

- **Shared Memory**: Increase `--shm-size=4g` in run script for large images
- **GPU**: Add `--gpus all` if NVIDIA GPU available
- **Batch**: Process multiple images by calling recognition model in batches

## Author

**Shivashankarar**  
Email: shivashankarar@ai4mtech.com

## License

Based on PaddleOCR (Apache 2.0) and NVIDIA Triton Server.
