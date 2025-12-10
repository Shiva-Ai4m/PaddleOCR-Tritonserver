#!/usr/bin/env python3
"""Complete OCR with correct PaddleOCR preprocessing and postprocessing"""

import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import json
import sys
import math
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

    # PP-OCRv3 English model outputs 97 classes, so add padding if needed
    while len(dict_character) < 97:
        dict_character.append('')  # Empty string for unknown characters

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
    """Calculate box score using fast method"""
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

def boxes_from_bitmap(pred, bitmap, src_w, src_h, thresh=0.3, box_thresh=0.7,
                     max_candidates=1000, unclip_ratio=2.0, min_size=3):
    """Extract boxes from bitmap - PaddleOCR DBPostProcess style"""
    height, width = bitmap.shape

    outs = cv2.findContours(
        (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(outs) == 3:
        contours = outs[1]
    elif len(outs) == 2:
        contours = outs[0]

    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []

    for index in range(num_contours):
        contour = contours[index]
        points, sside = get_mini_boxes(contour)
        if sside < min_size:
            continue

        points = np.array(points)
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue

        # Unclip
        box = unclip(points, unclip_ratio)
        if len(box) > 1:
            continue
        box = np.array(box).reshape(-1, 1, 2)

        box, sside = get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box)

        # Scale from detection output size to original image size
        box[:, 0] = np.clip(np.round(box[:, 0] / width * src_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * src_h), 0, src_h)

        boxes.append(box.astype(np.int32))
        scores.append(score)

    return boxes, scores

def postprocess_detection(pred, src_h, src_w, ratio_h, ratio_w, thresh=0.3, box_thresh=0.7):
    """Post-process detection output - PaddleOCR style"""
    pred = pred[0, 0, :, :]  # Remove batch and channel dims
    segmentation = pred > thresh

    boxes, scores = boxes_from_bitmap(pred, segmentation, src_w, src_h,
                                      thresh=thresh, box_thresh=box_thresh)

    return boxes, scores

def get_rotate_crop_image(img, box):
    """Crop and rotate text region"""
    box = np.array(box, dtype=np.float32)

    img_crop_width = int(max(
        np.linalg.norm(box[0] - box[1]),
        np.linalg.norm(box[2] - box[3])
    ))
    img_crop_height = int(max(
        np.linalg.norm(box[0] - box[3]),
        np.linalg.norm(box[1] - box[2])
    ))

    pts_dst = np.array([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box, pts_dst)
    img_crop = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return img_crop

def resize_norm_img(img, image_shape=(3, 48, 320), padding=True):
    """
    PaddleOCR recognition preprocessing - resize_norm_img

    Args:
        img: input image (H, W, C)
        image_shape: (C, H, W) - typically (3, 48, 320)
        padding: whether to pad image

    Returns:
        padded normalized image (C, H, W)
        valid_ratio: ratio of actual content width to total width
    """
    imgC, imgH, imgW = image_shape
    h, w = img.shape[:2]

    if not padding:
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_w = imgW
    else:
        # Calculate resize width maintaining aspect ratio
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))

    resized_image = resized_image.astype('float32')

    # Normalize
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255  # HWC -> CHW

    # Important: PaddleOCR uses (x - 0.5) / 0.5 normalization
    resized_image -= 0.5
    resized_image /= 0.5

    # Pad to target width
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image

    valid_ratio = min(1.0, float(resized_w / imgW))

    return padding_im, valid_ratio

def decode_ctc(preds_idx, preds_prob, character, is_remove_duplicate=True):
    """
    CTC decoding - PaddleOCR style

    Args:
        preds_idx: argmax of predictions (batch_size, seq_len)
        preds_prob: max probabilities (batch_size, seq_len)
        character: character dictionary
        is_remove_duplicate: whether to remove duplicate consecutive characters

    Returns:
        list of (text, confidence) tuples
    """
    result_list = []
    ignored_tokens = [0]  # blank token

    batch_size = len(preds_idx)
    for batch_idx in range(batch_size):
        selection = np.ones(len(preds_idx[batch_idx]), dtype=bool)

        # Remove duplicates
        if is_remove_duplicate:
            selection[1:] = preds_idx[batch_idx][1:] != preds_idx[batch_idx][:-1]

        # Remove ignored tokens (blank)
        for ignored_token in ignored_tokens:
            selection &= preds_idx[batch_idx] != ignored_token

        # Get characters (with bounds checking)
        char_list = []
        for text_id in preds_idx[batch_idx][selection]:
            if text_id < len(character):
                char_list.append(character[text_id])
            else:
                # Handle out of range indices (shouldn't happen but model might predict it)
                print(f"Warning: Character index {text_id} out of range (dict size: {len(character)})", flush=True)
                char_list.append('')

        # Get confidences
        if preds_prob is not None:
            conf_list = preds_prob[batch_idx][selection]
        else:
            conf_list = [1] * len(selection)

        if len(conf_list) == 0:
            conf_list = [0]

        text = ''.join(char_list)
        confidence = np.mean(conf_list).tolist()

        result_list.append((text, confidence))

    return result_list

def draw_boxes(img, results, boxes):
    """Draw bounding boxes and text on image"""
    img_vis = img.copy()

    # Create a mapping of boxes to results
    result_dict = {}
    for r in results:
        key = tuple(tuple(map(int, pt)) for pt in r['box'])
        result_dict[key] = r

    for i, box in enumerate(boxes):
        box_np = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        key = tuple(tuple(map(int, pt)) for pt in box)

        # Check if this box has recognized text
        if key in result_dict:
            result = result_dict[key]
            text = result['text']

            # Draw box in green for recognized text
            cv2.polylines(img_vis, [box_np], True, (0, 255, 0), 2)

            # Get top-left position for text
            min_x = int(min([p[0] for p in box]))
            min_y = int(min([p[1] for p in box]))

            # Draw text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Draw background rectangle
            cv2.rectangle(img_vis,
                         (min_x, min_y - text_h - 10),
                         (min_x + text_w + 5, min_y),
                         (0, 255, 0), -1)

            # Draw text
            cv2.putText(img_vis, text,
                       (min_x, min_y - 5),
                       font, font_scale, (0, 0, 0), thickness)
        else:
            # Draw box in red for unrecognized detections
            cv2.polylines(img_vis, [box_np], True, (0, 0, 255), 2)

    return img_vis

def ocr_pipeline(image_path, output_path=None):
    """Complete OCR pipeline with correct preprocessing and postprocessing"""
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Could not load image")
        return

    print(f"Image shape: {img.shape}")

    # Create Triton client
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Step 1: Detection preprocessing
    print("\n[1/4] Preprocessing for detection...")
    det_input, (src_h, src_w, ratio_h, ratio_w) = preprocess_detection(img)
    det_input_batch = np.expand_dims(det_input, axis=0).astype(np.float32)
    det_input_batch = np.ascontiguousarray(det_input_batch)

    print(f"  Original size: {src_w}x{src_h}")
    print(f"  Resized to: {det_input_batch.shape[3]}x{det_input_batch.shape[2]}")

    # Step 2: Run detection
    print("[2/4] Running detection model...")
    inputs = [grpcclient.InferInput("x", list(det_input_batch.shape), "FP32")]
    inputs[0].set_data_from_numpy(det_input_batch)
    outputs = [grpcclient.InferRequestedOutput("sigmoid_0.tmp_0")]

    det_response = triton_client.infer("ocr_detection", inputs=inputs, outputs=outputs)
    det_output = det_response.as_numpy("sigmoid_0.tmp_0")

    # Step 3: Post-process detection
    print("[3/4] Post-processing detection...")
    boxes, scores = postprocess_detection(det_output, src_h, src_w, ratio_h, ratio_w)
    print(f"Found {len(boxes)} text regions")

    # Step 4: Recognition for each box
    print("[4/4] Running recognition...")
    results = []
    for i, box in enumerate(boxes):
        try:
            # Crop text region
            img_crop = get_rotate_crop_image(img, box)

            if img_crop.shape[0] < 5 or img_crop.shape[1] < 5:
                continue

            # Preprocess for recognition using PaddleOCR method
            rec_input, valid_ratio = resize_norm_img(img_crop, image_shape=(3, 48, 320))
            rec_input_batch = np.expand_dims(rec_input, axis=0).astype(np.float32)
            rec_input_batch = np.ascontiguousarray(rec_input_batch)

            # Run recognition
            inputs = [grpcclient.InferInput("x", list(rec_input_batch.shape), "FP32")]
            inputs[0].set_data_from_numpy(rec_input_batch)
            outputs = [grpcclient.InferRequestedOutput("softmax_2.tmp_0")]

            rec_response = triton_client.infer("ocr_recognition", inputs=inputs, outputs=outputs)
            rec_output = rec_response.as_numpy("softmax_2.tmp_0")

            # CTC decoding using PaddleOCR method
            preds_idx = rec_output.argmax(axis=2)
            preds_prob = rec_output.max(axis=2)

            decoded = decode_ctc(preds_idx, preds_prob, CHARACTER, is_remove_duplicate=True)
            text, confidence = decoded[0]

            if text:
                results.append({
                    "text": text,
                    "box": box.tolist(),
                    "confidence": float(confidence),
                    "score": float(scores[i])
                })
                print(f"  Box {i+1}: '{text}' (conf: {confidence:.3f}, score: {scores[i]:.3f})")

        except Exception as e:
            print(f"  Box {i+1}: Error - {e}")
            import traceback
            traceback.print_exc()
            continue

    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"Total detections: {len(boxes)}")
    print(f"Successfully recognized: {len(results)} texts")

    if results:
        print("\nRecognized texts:")
        for i, item in enumerate(results, 1):
            print(f"  {i}. '{item['text']}'")

    # Save visualization if output path provided
    if output_path:
        print("\n[5/5] Creating visualization...")
        img_vis = draw_boxes(img, results, boxes)
        cv2.imwrite(output_path, img_vis)
        print(f"✓ Visualization saved to: {output_path}")

        # Save JSON results
        json_output = output_path.replace('.jpg', '.json').replace('.png', '.json')
        with open(json_output, 'w') as f:
            json.dump({
                'num_detections': len(boxes),
                'results': results,
                'image_size': {'width': src_w, 'height': src_h}
            }, f, indent=2)
        print(f"✓ JSON results saved to: {json_output}")

    print("="*60)
    print("\n✓ OCR completed successfully!")

    return results

if __name__ == "__main__":
    import os

    if len(sys.argv) < 2:
        print("Usage: python test_ocr_final_correct.py <input_image> [output_image]")
        print("Example: python test_ocr_final_correct.py screenshot_result.jpg output.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else image_path.replace('.jpg', '_correct_ocr.jpg').replace('.png', '_correct_ocr.jpg')

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    ocr_pipeline(image_path, output_path)
