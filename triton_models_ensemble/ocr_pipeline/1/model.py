import json
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils
from typing import List, Tuple

class TritonPythonModel:
    def initialize(self, args):
        print("[OCR Pipeline] Model initialized", flush=True)

        # Character dictionary for recognition
        self.character = [
            "blank", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "[", "\\", "]", "^", "_", "`",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "{", "|", "}", "~"
        ]

    def preprocess_detection(self, img: np.ndarray, limit_side_len: int = 960) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Preprocess image for detection model"""
        h, w = img.shape[:2]

        # Resize image
        ratio = 1.0
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # Round to 32
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        img_resized = cv2.resize(img, (resize_w, resize_h))

        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_normalized -= np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        img_normalized /= np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

        # Transpose to CHW
        img_transposed = img_normalized.transpose((2, 0, 1))

        ratio_h = resize_h / h
        ratio_w = resize_w / w

        return img_transposed, (ratio_h, ratio_w)

    def postprocess_detection(self, pred: np.ndarray, ratio_h: float, ratio_w: float,
                              thresh: float = 0.3, box_thresh: float = 0.6,
                              max_candidates: int = 1000, unclip_ratio: float = 1.5) -> List[List[int]]:
        """Post-process detection output to get bounding boxes"""

        # Get binary mask
        pred_map = pred[0, 0, :, :]  # Shape: (H, W)
        mask = pred_map > thresh

        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours[:max_candidates]:
            if contour.shape[0] < 4:
                continue

            # Get bounding box score
            points = contour.reshape(-1, 2)
            score = pred_map[points[:, 1].astype(int), points[:, 0].astype(int)].mean()

            if score < box_thresh:
                continue

            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box).astype(np.int32)

            # Unclip (expand) the box
            box = self.unclip_box(box, unclip_ratio)

            # Scale back to original image coordinates
            box[:, 0] = box[:, 0] / ratio_w
            box[:, 1] = box[:, 1] / ratio_h

            # Clip to image bounds and convert to int
            box = box.astype(np.int32).tolist()
            boxes.append(box)

        return boxes

    def unclip_box(self, box: np.ndarray, unclip_ratio: float) -> np.ndarray:
        """Expand the box by unclip_ratio"""
        try:
            from shapely.geometry import Polygon
            import pyclipper

            poly = Polygon(box)
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)

            if len(expanded) > 0:
                expanded_box = np.array(expanded[0])
                return expanded_box
        except:
            pass

        return box

    def preprocess_recognition(self, img_crop: np.ndarray, target_h: int = 48, target_w: int = 320) -> np.ndarray:
        """Preprocess cropped text region for recognition"""
        h, w = img_crop.shape[:2]

        # Resize maintaining aspect ratio
        ratio = target_h / h
        resize_w = int(w * ratio)

        if resize_w > target_w:
            resize_w = target_w

        img_resized = cv2.resize(img_crop, (resize_w, target_h))

        # Pad to target width
        if resize_w < target_w:
            img_padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            img_padded[:, :resize_w, :] = img_resized
            img_resized = img_padded

        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_normalized -= np.array([0.5, 0.5, 0.5]).reshape((1, 1, 3))
        img_normalized /= np.array([0.5, 0.5, 0.5]).reshape((1, 1, 3))

        # Transpose to CHW
        img_transposed = img_normalized.transpose((2, 0, 1))

        return img_transposed

    def decode_recognition(self, preds: np.ndarray) -> List[str]:
        """Decode recognition predictions"""
        texts = []

        for pred in preds:
            # pred shape: (seq_len, vocab_size)
            char_indices = np.argmax(pred, axis=1)

            # CTC decoding: remove blanks and duplicates
            text = []
            last_char = -1
            for idx in char_indices:
                if idx > 0 and idx != last_char:  # 0 is blank
                    if idx < len(self.character):
                        text.append(self.character[idx])
                last_char = idx

            texts.append(''.join(text))

        return texts

    def get_rotate_crop_image(self, img: np.ndarray, box: List[List[int]]) -> np.ndarray:
        """Crop and rotate text region from image"""
        box = np.array(box, dtype=np.float32)

        # Get width and height
        img_crop_width = int(max(
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[2] - box[3])
        ))
        img_crop_height = int(max(
            np.linalg.norm(box[0] - box[3]),
            np.linalg.norm(box[1] - box[2])
        ))

        # Define destination points
        pts_dst = np.array([
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height]
        ], dtype=np.float32)

        # Get perspective transform and apply
        M = cv2.getPerspectiveTransform(box, pts_dst)
        img_crop = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return img_crop

    def execute(self, requests):
        print(f"[OCR Pipeline] Received {len(requests)} requests", flush=True)
        responses = []

        for request in requests:
            try:
                # Get input image (already decoded as HWC numpy array)
                in_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
                img = in_tensor.as_numpy()

                print(f"[OCR Pipeline] Input image shape: {img.shape}, dtype: {img.dtype}", flush=True)

                if img.ndim != 3 or img.shape[2] != 3:
                    raise ValueError(f"Expected 3D image with 3 channels, got shape {img.shape}")

                orig_h, orig_w = img.shape[:2]
                print(f"[OCR Pipeline] Processing image: {orig_w}x{orig_h}", flush=True)

                # Step 1: Detection preprocessing
                det_input, (ratio_h, ratio_w) = self.preprocess_detection(img)
                det_input_batch = np.expand_dims(det_input, axis=0).astype(np.float32)

                # Ensure contiguous array for Triton
                det_input_batch = np.ascontiguousarray(det_input_batch)

                print(f"[OCR Pipeline] Detection input shape: {det_input_batch.shape}, dtype: {det_input_batch.dtype}", flush=True)
                print(f"[OCR Pipeline] Detection input contiguous: {det_input_batch.flags['C_CONTIGUOUS']}, size: {det_input_batch.nbytes} bytes", flush=True)

                # Step 2: Run detection model
                det_input_tensor = pb_utils.Tensor("x", det_input_batch)
                print(f"[OCR Pipeline] Created detection tensor, byte size: {det_input_tensor.as_numpy().nbytes}", flush=True)
                det_request = pb_utils.InferenceRequest(
                    model_name="ocr_detection",
                    requested_output_names=["sigmoid_0.tmp_0"],
                    inputs=[det_input_tensor]
                )

                det_response = det_request.exec()
                if det_response.has_error():
                    raise ValueError(f"Detection inference failed: {det_response.error().message()}")

                det_output = pb_utils.get_output_tensor_by_name(det_response, "sigmoid_0.tmp_0").as_numpy()
                print(f"[OCR Pipeline] Detection output shape: {det_output.shape}", flush=True)

                # Step 3: Post-process detection to get boxes
                boxes = self.postprocess_detection(det_output, ratio_h, ratio_w)
                print(f"[OCR Pipeline] Found {len(boxes)} text regions", flush=True)

                # Step 4: Recognition for each box
                results = []
                for i, box in enumerate(boxes):
                    try:
                        # Crop text region
                        img_crop = self.get_rotate_crop_image(img, box)

                        if img_crop.shape[0] < 5 or img_crop.shape[1] < 5:
                            continue

                        # Preprocess for recognition
                        rec_input = self.preprocess_recognition(img_crop)
                        rec_input_batch = np.expand_dims(rec_input, axis=0).astype(np.float32)

                        # Ensure contiguous array for Triton
                        rec_input_batch = np.ascontiguousarray(rec_input_batch)

                        # Run recognition model
                        rec_input_tensor = pb_utils.Tensor("x", rec_input_batch)
                        rec_request = pb_utils.InferenceRequest(
                            model_name="ocr_recognition",
                            requested_output_names=["softmax_2.tmp_0"],
                            inputs=[rec_input_tensor]
                        )

                        rec_response = rec_request.exec()
                        if rec_response.has_error():
                            print(f"[OCR Pipeline] Recognition failed for box {i}: {rec_response.error().message()}", flush=True)
                            continue

                        rec_output = pb_utils.get_output_tensor_by_name(rec_response, "softmax_2.tmp_0").as_numpy()

                        # Decode text
                        texts = self.decode_recognition(rec_output)
                        if texts and texts[0]:
                            results.append({
                                "text": texts[0],
                                "box": box,
                                "confidence": float(rec_output.max())
                            })
                            print(f"[OCR Pipeline] Box {i}: '{texts[0]}'", flush=True)

                    except Exception as e:
                        print(f"[OCR Pipeline] Error processing box {i}: {e}", flush=True)
                        continue

                # Create response JSON
                result = json.dumps({
                    "num_detections": len(boxes),
                    "results": results,
                    "image_size": {"width": orig_w, "height": orig_h}
                })

                print(f"[OCR Pipeline] Sending {len(results)} results", flush=True)

                # Create output tensor
                results_list = [result]
                result_array = np.array(
                    list(map(lambda x: x.encode("utf-8"), results_list)),
                    dtype=np.object_
                ).reshape(len(results_list), -1)

                out_tensor = pb_utils.Tensor("OUTPUT", result_array)
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)

            except Exception as e:
                print(f"[OCR Pipeline] Error: {e}", flush=True)
                import traceback
                traceback.print_exc()

                # Return error response
                error_result = json.dumps({"error": str(e), "results": []})
                result_array = np.array([error_result.encode("utf-8")], dtype=np.object_).reshape(1, -1)
                out_tensor = pb_utils.Tensor("OUTPUT", result_array)
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)

        return responses

    def finalize(self):
        print("[OCR Pipeline] Model finalized", flush=True)
