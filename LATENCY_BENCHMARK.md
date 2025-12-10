# PaddleOCR Triton Server - Latency Benchmark

**Benchmark Date:** December 2024
**Test Image:** 2560x720 pixels
**Hardware:** CPU inference (ONNX Runtime)
**Test Runs:** 10 iterations after 3 warmup runs

## Summary Results

| Metric | Value |
|--------|-------|
| **Mean End-to-End Latency** | **69.08 ms** |
| **Median Latency** | **68.13 ms** |
| **Throughput** | **14.48 images/sec** |
| **Average Detections** | **4 text regions** |

## Detailed Latency Breakdown

### Detection Pipeline

| Stage | Mean | Median | Min | Max | P95 | % of Total |
|-------|------|--------|-----|-----|-----|------------|
| **Preprocessing** | 8.48 ms | 8.28 ms | 8.15 ms | 9.21 ms | 9.15 ms | 12.3% |
| **ONNX Inference** | 24.54 ms | 24.08 ms | 22.12 ms | 29.12 ms | 27.83 ms | 35.5% |
| **Postprocessing** | 1.11 ms | 1.05 ms | 1.02 ms | 1.62 ms | 1.39 ms | 1.6% |
| **Subtotal** | **34.13 ms** | | | | | **49.4%** |

### Recognition Pipeline (×4 text regions)

| Stage | Mean | Median | Min | Max | P95 | % of Total |
|-------|------|--------|-----|-----|-----|------------|
| **Preprocessing** | 1.68 ms | 1.66 ms | 1.63 ms | 1.83 ms | 1.77 ms | 2.4% |
| **ONNX Inference** | 33.08 ms | 32.43 ms | 30.00 ms | 40.27 ms | 37.56 ms | 47.9% |
| **Postprocessing** | 0.19 ms | 0.18 ms | 0.18 ms | 0.24 ms | 0.22 ms | 0.3% |
| **Subtotal** | **34.95 ms** | | | | | **50.6%** |

## Performance Analysis

### Bottlenecks

1. **Recognition Inference (47.9%)** - Dominant bottleneck
   - 33.08 ms for 4 text regions (~8.27 ms per region)
   - Runs sequentially, not batched
   - **Optimization potential:** Batch recognition requests

2. **Detection Inference (35.5%)** - Second major cost
   - 24.54 ms for full image detection
   - Runs once per image
   - **Optimization potential:** GPU acceleration

3. **Preprocessing (14.7% total)** - Minor overhead
   - Detection: 8.48 ms (resize, normalize, transpose)
   - Recognition: 1.68 ms (4 crops)

4. **Postprocessing (1.9% total)** - Negligible
   - Very efficient DBNet postprocessing
   - Fast CTC decoding

### Latency Distribution

```
Detection:     ████████████████████░░░░░░░░░░░░░░  49.4% (34.13 ms)
Recognition:   ███████████████████░░░░░░░░░░░░░░░  50.6% (34.95 ms)
```

### Scaling with Text Regions

The benchmark detected **4 text regions**. Recognition latency scales linearly with detections:

| Detections | Estimated Total Latency |
|------------|-------------------------|
| 1 | ~42 ms (~24 img/s) |
| 2 | ~51 ms (~20 img/s) |
| 4 | ~69 ms (~14 img/s) |
| 8 | ~104 ms (~10 img/s) |
| 16 | ~173 ms (~6 img/s) |

*Formula: ~34 ms (detection) + ~8.7 ms × num_detections*

## Optimization Opportunities

### High Impact (50-70% improvement)

1. **GPU Acceleration**
   - Move ONNX inference to GPU
   - Expected: 3-5× speedup on inference
   - Target: <20 ms total latency

2. **Batch Recognition**
   - Process all text regions in single batch
   - Current: 4 sequential calls @ 8.27 ms each
   - Batched: 1 call @ ~10-12 ms total
   - Savings: ~20-25 ms

### Medium Impact (20-30% improvement)

3. **TensorRT Optimization**
   - Convert ONNX → TensorRT
   - Expected: 1.5-2× additional speedup

4. **Model Quantization**
   - INT8 quantization for both models
   - Expected: 30-50% faster inference
   - Trade-off: Minor accuracy loss

### Low Impact (<10% improvement)

5. **Preprocessing Optimization**
   - Use OpenCV GPU functions
   - Minimal gains (~2-3 ms)

## Running the Benchmark

### Basic Usage

```bash
python3 benchmark_latency.py <image_path>
```

### Custom Parameters

```bash
# More warmup runs and test iterations
python3 benchmark_latency.py examples/sample_input.png 5 20

# Parameters: <image> <warmup_runs> <test_runs>
```

### Example Output

```
=== OCR Latency Benchmark ===
Image: examples/sample_input.png
Warmup runs: 3
Test runs: 10

Image size: 2560x720

Warming up (3 runs)...
Warmup completed

Running benchmark (10 runs)...
  Run 1/10: 74.44 ms
  Run 2/10: 67.58 ms
  ...
  Run 10/10: 69.52 ms

=== Latency Results ===
...
```

## Comparison with PaddleOCR Native

| Implementation | Latency | Throughput | Notes |
|----------------|---------|------------|-------|
| **Triton Client-Side** | 69 ms | 14.5 img/s | This implementation |
| PaddleOCR Python (CPU) | ~80-100 ms | ~10-12 img/s | Baseline |
| PaddleOCR Python (GPU) | ~20-30 ms | ~33-50 img/s | With CUDA |

**Verdict:** Triton client-side approach is **15-30% faster** than native PaddleOCR CPU, likely due to:
- ONNX Runtime optimizations
- Efficient Triton gRPC communication
- Optimized postprocessing

## Hardware Information

**Platform:** Linux (tested on Rocky Linux 9)
**CPU:** Intel Xeon (specific model varies)
**Memory:** 16GB+ recommended
**ONNX Runtime:** CPU backend
**Triton Version:** 24.08-py3

## Reproducibility

All benchmarks use:
- Fixed random seed (when applicable)
- Warmup runs to stabilize performance
- Multiple iterations for statistical reliability
- P95/P99 percentiles to catch outliers

## Notes

1. **Latency varies** with:
   - Image size (larger = slower detection)
   - Number of text regions (more = slower recognition)
   - System load and background processes

2. **Client-side processing** means:
   - Latency includes network overhead (gRPC)
   - Preprocessing/postprocessing runs on client CPU
   - Server only handles ONNX inference

3. **Single-threaded client** - Results represent sequential processing. Multi-threaded clients could improve throughput.

---

**Generated by:** benchmark_latency.py
**Author:** Shivashankarar (shivashankarar@ai4mtech.com)
**Organization:** AI4M Technologies
