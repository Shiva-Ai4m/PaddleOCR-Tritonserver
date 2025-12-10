# PaddleOCR Triton Inference Server

**Status:** ✅ Working (Client-side Processing)
**Architecture:** Client-side preprocessing/postprocessing + Server ONNX inference

## Overview

OCR implementation using NVIDIA Triton Inference Server with PaddleOCR PP-OCRv3 models. Due to Triton BLS data transmission bugs, preprocessing and postprocessing are done client-side, while the server handles ONNX model inference.

## Architecture

```
Client (Python)                    Triton Server
---------------                    -------------
1. Load image
2. Detection preprocessing    -->  [ocr_detection (ONNX)]  -->  Detection map
3. Detection postprocessing
4. Crop text regions
5. Recognition preprocessing  -->  [ocr_recognition (ONNX)] -->  CTC predictions  
6. Recognition postprocessing
7. Return results
```

## Features

- ✅ **Complete OCR Pipeline**: Text detection + recognition
- ✅ **PaddleOCR PP-OCRv3**: Industry-standard English OCR models
- ✅ **ONNX Inference**: Fast inference with ONNX Runtime
- ✅ **Accurate**: Implements PaddleOCR's exact algorithms

## Models

```
triton_models_ensemble/
├── ocr_detection/          # Detection ONNX model
│   ├── 1/en_PP-OCRv3_det.onnx
│   └── config.pbtxt
│
├── ocr_recognition/        # Recognition ONNX model
│   ├── 1/en_PP-OCRv3_rec.onnx
│   └── config.pbtxt
│
└── ocr_pipeline/          # Python BLS (NOT WORKING - Triton bug)
    └── (included but not functional)
```

## Setup

### 1. Install Dependencies

```bash
pip install tritonclient[grpc] opencv-python numpy shapely pyclipper
```

### 2. Start Triton Server

```bash
./run_triton_ensemble.sh
```

Wait ~30 seconds for models to load.

### 3. Verify Server

```bash
curl localhost:8000/v2/health/ready
# Should return: {"ready":true}
```

## Usage

### Run OCR

```bash
python3 test_ocr_final_correct.py <input_image> [output_visualization.jpg]
```

### Examples

```bash
# Basic usage
python3 test_ocr_final_correct.py screenshot.png

# With visualization output
python3 test_ocr_final_correct.py document.jpg result.jpg
```

## Output

The script generates:
1. **Visualization**: Image with green bounding boxes and recognized text
2. **JSON file**: Structured results with bboxes, text, and confidence scores

```json
[
  {
    "bbox": [[10, 20], [100, 20], [100, 50], [10, 50]],
    "text": "Hello World",
    "rec_conf": 0.987,
    "det_score": 0.942
  }
]
```

## Known Limitations

1. **Client-side Processing**: Preprocessing/postprocessing done on client (less scalable)
2. **BLS Not Working**: ocr_pipeline model exists but has Triton data transmission bugs
3. **Fixed Width**: Recognition limited to 320px width per text region

## Why Client-Side?

Triton Server 24.08 has a bug where large tensors (>2MB) fail to transmit between BLS models, receiving 0 bytes instead. This affects:
- Server-side preprocessing → ONNX
- ONNX → server-side postprocessing  
- Python BLS orchestration

**Solution**: Keep heavy data processing on client side, use server only for ONNX inference.

## Troubleshooting

### Server Not Starting

Check logs:
```bash
podman logs triton-ocr
```

### Connection Refused

Ensure server is running:
```bash
podman ps | grep triton-ocr
curl localhost:8000/v2/health/ready
```

## Commands

```bash
# Start server
./run_triton_ensemble.sh

# Run OCR
python3 test_ocr_final_correct.py image.jpg output.jpg

# View server logs
podman logs -f triton-ocr

# Stop server
podman stop triton-ocr
```

## License

Based on PaddleOCR (Apache 2.0) and NVIDIA Triton Server.

## References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [PP-OCRv3 Paper](https://arxiv.org/abs/2206.03001)
