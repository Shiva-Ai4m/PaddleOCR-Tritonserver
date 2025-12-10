#!/usr/bin/env python3
"""Benchmark OCR latency for single image processing"""

import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import json
import sys
import time
from shapely.geometry import Polygon
import pyclipper

# Load character dictionary (PaddleOCR en_dict.txt)
def load_character_dict(dict_path="../ppocr/utils/en_dict.txt"):
    """Load character dictionary from file"""
    character_str = []
    with open(dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode("utf-8").strip("\n").strip("\r\n")
            character_str.append(line)

    # Add blank at the beginning for CTC
    dict_character = ["blank"] + character_str

    # PP-OCRv3 English model outputs 97 classes
    while len(dict_character) < 97:
        dict_character.append('')

    return dict_character

# Load the character dictionary
CHARACTER = load_character_dict()

def preprocess_detection(img, limit_side_len=960):
    """Preprocess image for detection - PaddleOCR style"""
    src_h, src_w, _ = img.shape

    # Resize keeping aspect ratio
    ratio = 1.0
    if max(src_h, src_w) > limit_side_len:
        if src_h > src_w:
            ratio = float(limit_side_len) / src_h
        else:
            ratio = float(limit_side_len) / src_w

    resize_h = int(src_h * ratio)
    resize_w = int(src_w * ratio)

    # Round to multiple of 32
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    img_resized = cv2.resize(img, (resize_w, resize_h))

    ratio_h = float(resize_h) / src_h
    ratio_w = float(resize_w) / src_w

    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized -= np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    img_normalized /= np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    # Transpose to CHW
    img_transposed = img_normalized.transpose((2, 0, 1))

    return img_transposed, (src_h, src_w, ratio_h, ratio_w)

def get_mini_boxes(contour):
    """Get minimum bounding box with correct point ordering"""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])

def unclip(box, unclip_ratio=2.0):
    """Expand box using pyclipper"""
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return expanded

def box_score_fast(bitmap, _box):
    """Calculate average score in box region"""
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

def postprocess_detection(pred, shape_info, box_thresh=0.5, unclip_ratio=2.0):
    """Postprocess detection output - PaddleOCR DBPostProcess"""
    src_h, src_w, ratio_h, ratio_w = shape_info

    # Get binary mask
    bitmap = pred[0, 0, :, :]
    pred_mask = bitmap > box_thresh

    # Find contours
    contours, _ = cv2.findContours(
        (pred_mask * 255).astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    scores = []

    for contour in contours:
        if contour.shape[0] <= 2:
            continue

        # Get minimum bounding box
        box, sside = get_mini_boxes(contour)
        if sside < 3:
            continue

        box = np.array(box)

        # Calculate score
        score = box_score_fast(bitmap, box)
        if score < box_thresh:
            continue

        # Unclip box
        expanded = unclip(box, unclip_ratio)
        if len(expanded) == 0:
            continue

        expanded = np.array(expanded[0]).reshape(-1, 2)
        if expanded.shape[0] <= 2:
            continue

        exp_box, exp_sside = get_mini_boxes(expanded)
        if exp_sside < 3:
            continue

        exp_box = np.array(exp_box)

        # Scale back to original image
        exp_box[:, 0] = np.clip(exp_box[:, 0] / ratio_w, 0, src_w)
        exp_box[:, 1] = np.clip(exp_box[:, 1] / ratio_h, 0, src_h)

        boxes.append(exp_box.astype(int).tolist())
        scores.append(float(score))

    return boxes, scores

def preprocess_recognition(img, img_height=48, max_width=320):
    """Preprocess cropped image for recognition"""
    h, w = img.shape[:2]

    # Calculate resize ratio
    ratio = float(h) / img_height
    resize_w = int(w / ratio)

    # Limit width
    if resize_w > max_width:
        resize_w = max_width

    # Resize
    img_resized = cv2.resize(img, (resize_w, img_height))

    # Pad to max_width
    if resize_w < max_width:
        img_padded = np.zeros((img_height, max_width, 3), dtype=np.float32)
        img_padded[:, :resize_w, :] = img_resized
        img_resized = img_padded

    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5

    # Transpose to CHW
    img_transposed = img_normalized.transpose((2, 0, 1))

    return img_transposed

def postprocess_recognition(preds):
    """Decode CTC output"""
    preds_idx = preds.argmax(axis=2)
    preds_prob = preds.max(axis=2)

    results = []
    for pred_idx, pred_prob in zip(preds_idx, preds_prob):
        # CTC decode
        char_list = []
        conf_list = []
        last_idx = -1

        for idx, prob in zip(pred_idx, pred_prob):
            if idx == 0 or idx == last_idx:  # blank or repeat
                last_idx = idx
                continue

            char_list.append(CHARACTER[idx])
            conf_list.append(float(prob))
            last_idx = idx

        text = ''.join(char_list)
        confidence = float(np.mean(conf_list)) if conf_list else 0.0
        results.append((text, confidence))

    return results

def order_points_clockwise(pts):
    """Order points in clockwise order starting from top-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_text_region(img, box):
    """Crop and perspective transform text region"""
    box = np.array(box, dtype=np.float32)
    box = order_points_clockwise(box)

    width = int(max(
        np.linalg.norm(box[0] - box[1]),
        np.linalg.norm(box[2] - box[3])
    ))
    height = int(max(
        np.linalg.norm(box[0] - box[3]),
        np.linalg.norm(box[1] - box[2])
    ))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped

def benchmark_ocr(image_path, warmup_runs=3, test_runs=10):
    """Benchmark OCR latency"""
    print(f"=== OCR Latency Benchmark ===")
    print(f"Image: {image_path}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Test runs: {test_runs}")
    print()

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Connect to Triton
    try:
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
    except Exception as e:
        print(f"Error: Could not connect to Triton server: {e}")
        return

    # Warmup runs
    print(f"\nWarming up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        # Detection preprocessing
        det_input, shape_info = preprocess_detection(img)
        det_input = np.expand_dims(det_input, axis=0).astype(np.float32)

        # Detection inference
        inputs = [grpcclient.InferInput("x", det_input.shape, "FP32")]
        inputs[0].set_data_from_numpy(det_input)
        outputs = [grpcclient.InferRequestedOutput("sigmoid_0.tmp_0")]
        response = triton_client.infer("ocr_detection", inputs, outputs=outputs)
        det_output = response.as_numpy("sigmoid_0.tmp_0")

        # Detection postprocessing
        boxes, scores = postprocess_detection(det_output, shape_info)

        if boxes:
            # Crop and recognize first box
            cropped = crop_text_region(img, boxes[0])
            rec_input = preprocess_recognition(cropped)
            rec_input = np.expand_dims(rec_input, axis=0).astype(np.float32)

            inputs = [grpcclient.InferInput("x", rec_input.shape, "FP32")]
            inputs[0].set_data_from_numpy(rec_input)
            outputs = [grpcclient.InferRequestedOutput("softmax_2.tmp_0")]
            response = triton_client.infer("ocr_recognition", inputs, outputs=outputs)

    print("Warmup completed\n")

    # Benchmark runs
    print(f"Running benchmark ({test_runs} runs)...")

    timings = {
        "total": [],
        "detection_preprocess": [],
        "detection_inference": [],
        "detection_postprocess": [],
        "recognition_preprocess": [],
        "recognition_inference": [],
        "recognition_postprocess": []
    }

    num_detections_list = []

    for run in range(test_runs):
        t_start = time.perf_counter()

        # Detection preprocessing
        t0 = time.perf_counter()
        det_input, shape_info = preprocess_detection(img)
        det_input = np.expand_dims(det_input, axis=0).astype(np.float32)
        t1 = time.perf_counter()
        timings["detection_preprocess"].append((t1 - t0) * 1000)

        # Detection inference
        t0 = time.perf_counter()
        inputs = [grpcclient.InferInput("x", det_input.shape, "FP32")]
        inputs[0].set_data_from_numpy(det_input)
        outputs = [grpcclient.InferRequestedOutput("sigmoid_0.tmp_0")]
        response = triton_client.infer("ocr_detection", inputs, outputs=outputs)
        det_output = response.as_numpy("sigmoid_0.tmp_0")
        t1 = time.perf_counter()
        timings["detection_inference"].append((t1 - t0) * 1000)

        # Detection postprocessing
        t0 = time.perf_counter()
        boxes, scores = postprocess_detection(det_output, shape_info)
        t1 = time.perf_counter()
        timings["detection_postprocess"].append((t1 - t0) * 1000)

        num_detections_list.append(len(boxes))

        # Recognition (all detected boxes)
        t_rec_pre_total = 0
        t_rec_inf_total = 0
        t_rec_post_total = 0

        for box in boxes:
            # Recognition preprocessing
            t0 = time.perf_counter()
            cropped = crop_text_region(img, box)
            rec_input = preprocess_recognition(cropped)
            rec_input = np.expand_dims(rec_input, axis=0).astype(np.float32)
            t1 = time.perf_counter()
            t_rec_pre_total += (t1 - t0) * 1000

            # Recognition inference
            t0 = time.perf_counter()
            inputs = [grpcclient.InferInput("x", rec_input.shape, "FP32")]
            inputs[0].set_data_from_numpy(rec_input)
            outputs = [grpcclient.InferRequestedOutput("softmax_2.tmp_0")]
            response = triton_client.infer("ocr_recognition", inputs, outputs=outputs)
            rec_output = response.as_numpy("softmax_2.tmp_0")
            t1 = time.perf_counter()
            t_rec_inf_total += (t1 - t0) * 1000

            # Recognition postprocessing
            t0 = time.perf_counter()
            results = postprocess_recognition(rec_output)
            t1 = time.perf_counter()
            t_rec_post_total += (t1 - t0) * 1000

        timings["recognition_preprocess"].append(t_rec_pre_total)
        timings["recognition_inference"].append(t_rec_inf_total)
        timings["recognition_postprocess"].append(t_rec_post_total)

        t_end = time.perf_counter()
        timings["total"].append((t_end - t_start) * 1000)

        print(f"  Run {run+1}/{test_runs}: {timings['total'][-1]:.2f} ms")

    # Calculate statistics
    print("\n=== Latency Results ===\n")

    avg_detections = np.mean(num_detections_list)
    print(f"Average detections per image: {avg_detections:.1f}\n")

    def print_stats(name, values):
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        p50 = np.percentile(values, 50)
        p95 = np.percentile(values, 95)
        p99 = np.percentile(values, 99)

        print(f"{name}:")
        print(f"  Mean:   {mean:8.2f} ms  (±{std:.2f} ms)")
        print(f"  Median: {p50:8.2f} ms")
        print(f"  Min:    {min_val:8.2f} ms")
        print(f"  Max:    {max_val:8.2f} ms")
        print(f"  P95:    {p95:8.2f} ms")
        print(f"  P99:    {p99:8.2f} ms")
        print()

    print_stats("Total End-to-End Latency", timings["total"])
    print_stats("Detection Preprocessing", timings["detection_preprocess"])
    print_stats("Detection Inference (ONNX)", timings["detection_inference"])
    print_stats("Detection Postprocessing", timings["detection_postprocess"])
    print_stats(f"Recognition Preprocessing (×{avg_detections:.0f})", timings["recognition_preprocess"])
    print_stats(f"Recognition Inference (×{avg_detections:.0f})", timings["recognition_inference"])
    print_stats(f"Recognition Postprocessing (×{avg_detections:.0f})", timings["recognition_postprocess"])

    # Breakdown
    print("=== Latency Breakdown (Mean) ===\n")
    total_mean = np.mean(timings["total"])
    det_pre_mean = np.mean(timings["detection_preprocess"])
    det_inf_mean = np.mean(timings["detection_inference"])
    det_post_mean = np.mean(timings["detection_postprocess"])
    rec_pre_mean = np.mean(timings["recognition_preprocess"])
    rec_inf_mean = np.mean(timings["recognition_inference"])
    rec_post_mean = np.mean(timings["recognition_postprocess"])

    print(f"Detection Preprocessing:   {det_pre_mean:8.2f} ms ({det_pre_mean/total_mean*100:5.1f}%)")
    print(f"Detection Inference:       {det_inf_mean:8.2f} ms ({det_inf_mean/total_mean*100:5.1f}%)")
    print(f"Detection Postprocessing:  {det_post_mean:8.2f} ms ({det_post_mean/total_mean*100:5.1f}%)")
    print(f"Recognition Preprocessing: {rec_pre_mean:8.2f} ms ({rec_pre_mean/total_mean*100:5.1f}%)")
    print(f"Recognition Inference:     {rec_inf_mean:8.2f} ms ({rec_inf_mean/total_mean*100:5.1f}%)")
    print(f"Recognition Postprocessing:{rec_post_mean:8.2f} ms ({rec_post_mean/total_mean*100:5.1f}%)")
    print(f"{'─'*45}")
    print(f"Total:                     {total_mean:8.2f} ms (100.0%)")
    print()

    # Throughput
    throughput = 1000.0 / total_mean
    print(f"=== Throughput ===\n")
    print(f"Images per second: {throughput:.2f} img/s")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark_latency.py <image_path> [warmup_runs] [test_runs]")
        print("\nExample:")
        print("  python3 benchmark_latency.py examples/sample_input.png")
        print("  python3 benchmark_latency.py examples/sample_input.png 5 20")
        sys.exit(1)

    image_path = sys.argv[1]
    warmup_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    test_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    benchmark_ocr(image_path, warmup_runs, test_runs)
