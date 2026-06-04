"""
ai/detector/person_detector.py
────────────────────────────────
Person Detector — wraps YOLO (and HOG fallback) for person detection.

Architecture:
  Each frame goes through the detector to find bounding boxes around
  every person in the scene. The boxes are then passed to the tracker.

               ┌─────────────────────────────┐
  frame (BGR)  │      PersonDetector         │  detections
  ────────────▶│   YOLO / HOG backend        │──────────▶  list[Detection]
               └─────────────────────────────┘
                     ↑ swappable backend

Detector hierarchy (best to worst):
  1. YOLOv8n  — 6 MB model, very fast, high accuracy, GPU-optional
  2. HOG+SVM  — built into OpenCV, no model file needed, slower
  3. Mock      — returns synthetic detections (for CI / unit tests)

The detector is chosen automatically at startup based on what's available.
When you add a GPU server, simply installing `ultralytics` and placing
`yolov8n.pt` in models/ switches the whole pipeline to YOLO.

Detection output schema:
  Detection = {
    "bbox":       [x1, y1, x2, y2],   # pixels, absolute coords
    "confidence": float,               # 0.0–1.0
    "class_id":   int,                 # 0 = person (COCO)
    "class_name": str,                 # "person"
  }
"""

from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Detection dataclass ───────────────────────────────────────────────────────

@dataclass
class Detection:
    """One detected person bounding box."""
    bbox:       tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    class_id:   int   = 0
    class_name: str   = "person"

    @property
    def x1(self) -> int: return self.bbox[0]
    @property
    def y1(self) -> int: return self.bbox[1]
    @property
    def x2(self) -> int: return self.bbox[2]
    @property
    def y2(self) -> int: return self.bbox[3]

    @property
    def width(self)  -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    @property
    def area(self) -> int: return self.width * self.height

    def to_tlwh(self) -> tuple[int, int, int, int]:
        """Convert to (top, left, width, height) format — used by some trackers."""
        return (self.x1, self.y1, self.width, self.height)

    def iou(self, other: "Detection") -> float:
        """Intersection-over-Union with another detection."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


# ── Base detector interface ───────────────────────────────────────────────────

class BaseDetector:
    """Abstract base — all detectors implement detect()."""

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str = "unknown",
    ) -> list[Detection]:
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: run inference on a dummy frame to initialise GPU kernels."""
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        self.detect(dummy)


# ── Backend 1: YOLOv8 ────────────────────────────────────────────────────────

class YOLODetector(BaseDetector):
    """
    YOLOv8 person detector via the Ultralytics library.

    Model sizes (speed vs accuracy):
      yolov8n.pt  →  6 MB   fastest, good for CPU
      yolov8s.pt  →  22 MB  better accuracy, still fast
      yolov8m.pt  →  52 MB  high accuracy, needs GPU
      yolov8l.pt  →  87 MB  best accuracy, GPU required

    Precision enhancements:
      • CLAHE preprocessing — improves detection in low-light warehouse scenes
      • Larger input size (960) — better detection of small/far persons
      • Test-time augmentation (optional flip ensemble)
      • Adaptive confidence thresholding based on bbox size
      • Soft-NMS with score re-weighting for overlapping detections
      • Aspect-ratio confidence boost (tall boxes = more person-like)
    """

    def __init__(
        self,
        model_path:        str   = "models/yolov8n.pt",
        confidence_thresh: float = 0.25,
        device:            str   = "cpu",
        img_size:          int   = 960,
        use_tta:           bool  = False,
        use_clahe:         bool  = True,
        debug_dir:         Optional[str] = None,
    ):
        from ultralytics import YOLO

        self.conf        = confidence_thresh
        self.iou_thresh  = 0.50
        self.img_size    = img_size
        self.device      = device
        self.use_tta     = use_tta
        self.use_clahe   = use_clahe
        self._debug_dir  = debug_dir
        self._debug_counter: dict[str, int] = {}

        logger.info(f"Loading YOLO model: {model_path} on {device}")

        # PyTorch >= 2.6 defaults weights_only=True which breaks YOLO model loading.
        # Patch torch.load temporarily to set weights_only=False.
        import torch
        _original_torch_load = torch.load
        def _patched_torch_load(f, *args, **kwargs):
            kwargs["weights_only"] = False
            return _original_torch_load(f, *args, **kwargs)
        torch.load = _patched_torch_load

        try:
            self.model = YOLO(model_path, task="detect")
        finally:
            torch.load = _original_torch_load

        self.model.to(device)

        if self.use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            logger.info("CLAHE preprocessing enabled")

        self._person_class = 0
        logger.info(f"YOLODetector ready | imgsz={img_size} | TTA={use_tta} | CLAHE={use_clahe}")

    COCO_NAMES: dict[int, str] = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
        34: "baseball bat", 35: "baseball glove", 36: "skateboard",
        37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
        41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
        51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
        56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
        60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
        64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
        68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
        72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
        76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
    }

    # Classes that could reasonably be carried by a person
    CARRYABLE_CLASSES: set[int] = {
        24,   # backpack
        26,   # handbag
        28,   # suitcase
        39,   # bottle
        40,   # wine glass
        41,   # cup
        63,   # laptop
        64,   # mouse
        65,   # remote
        66,   # keyboard
        67,   # cell phone
        73,   # book
        76,   # scissors
        77,   # teddy bear
        78,   # hair drier
        79,   # toothbrush
    }

    def _save_debug_stage(
        self,
        camera_id: str,
        stage:     str,
        frame_bgr: np.ndarray,
        detections: list[Detection],
    ) -> None:
        """Save a debug frame with detections drawn for intermediate stage analysis."""
        if not self._debug_dir:
            return
        self._debug_counter[camera_id] = self._debug_counter.get(camera_id, 0) + 1
        fn = self._debug_counter[camera_id]
        viz = frame_bgr.copy()
        for d in detections:
            cv2.rectangle(viz, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            cv2.putText(viz, f"{d.confidence:.2f}", (d.x1, max(d.y1-5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        import os
        save_dir = os.path.join(self._debug_dir, camera_id, stage)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"frame_{fn:06d}.jpg")
        cv2.imwrite(path, viz, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str = "unknown",
    ) -> list[Detection]:
        """
        Run YOLOv8 inference on a single BGR frame with precision enhancements.

        Args:
            frame: BGR numpy array (H × W × 3)
            camera_id: Camera identifier for per-camera logging.

        Returns:
            List of Detection objects, persons only, above confidence threshold.
        """
        t_start = time.perf_counter()

        # ── Step 1: Preprocess ──────────────────────────────────────────────────
        original = frame.copy()
        processed = self._preprocess(frame)

        # ── Step 2: Run inference ──────────────────────────────────────────────
        t_infer_start = time.perf_counter()
        results = self.model(
            processed,
            conf=    self.conf * 0.8,
            iou=     self.iou_thresh,
            classes= [self._person_class],
            imgsz=   self.img_size,
            verbose= False,
        )[0]
        t_infer = time.perf_counter() - t_infer_start

        raw_detections = self._parse_results(results)
        n_raw = len(raw_detections)

        logger.debug(
            f"[{camera_id}] YOLO inference: {t_infer*1000:.1f}ms | "
            f"raw_detections={n_raw} | frame_shape={frame.shape}"
        )
        for idx, d in enumerate(raw_detections):
            logger.debug(
                f"[{camera_id}]   Raw det #{idx}: "
                f"box=({d.x1},{d.y1},{d.x2},{d.y2}) "
                f"conf={d.confidence:.4f} area={d.area} "
                f"aspect={d.height/max(d.width,1):.2f}"
            )

        detections = list(raw_detections)

        # ── Step 2b: Log YOLO raw output ──────────────────────────────────────
        logger.info(
            f"🔍 DETECT[raw] camera={camera_id} "
            f"frame_h={processed.shape[0]} frame_w={processed.shape[1]} "
            f"raw_detections={n_raw} "
            f"confs=[{', '.join(f'{d.confidence:.3f}' for d in raw_detections)}] "
            f"bboxes=[{'; '.join(f'({d.x1},{d.y1},{d.x2},{d.y2})' for d in raw_detections)}] "
            f"areas=[{', '.join(str(d.area) for d in raw_detections)}]"
        )
        self._save_debug_stage(camera_id, "yolo_raw", original, raw_detections)

        # ── Step 3: Test-time augmentation (flip ensemble) ─────────────────────
        if self.use_tta:
            flipped = cv2.flip(processed, 1)
            results_f = self.model(
                flipped,
                conf=    self.conf * 0.8,
                iou=     self.iou_thresh,
                classes= [self._person_class],
                imgsz=   self.img_size,
                verbose= False,
            )[0]
            tta_dets = self._parse_results(results_f)
            h, w = processed.shape[:2]
            for d in tta_dets:
                x1, y1, x2, y2 = d.bbox
                d.bbox = (w - x2, y1, w - x1, y2)
            detections = self._merge_tta(detections, tta_dets)
            logger.debug(f"[{camera_id}]   After TTA merge: {len(detections)} detections")

        # ── Step 4a: Adaptive confidence filter ────────────────────────────────
        n_after_raw = len(detections)
        if detections:
            n_pre_adaptive = len(detections)
            detections = self._adaptive_filter(detections, frame.shape[:2], camera_id)
            n_post_adaptive = len(detections)
            filtered_by_adaptive = n_pre_adaptive - n_post_adaptive
            logger.info(
                f"🔍 DETECT[adaptive] camera={camera_id} "
                f"before={n_pre_adaptive} after={n_post_adaptive} "
                f"filtered={filtered_by_adaptive} "
                f"kept_confs=[{', '.join(f'{d.confidence:.3f}' for d in detections)}]"
            )

        self._save_debug_stage(camera_id, "after_adaptive", original, detections)

        # ── Step 4b: Soft-NMS ──────────────────────────────────────────────────
        n_after_adaptive = len(detections)
        if detections:
            n_pre_nms = len(detections)
            detections = self._soft_nms(detections, iou_threshold=self.iou_thresh)
            n_post_nms = len(detections)
            filtered_by_nms = n_pre_nms - n_post_nms
            logger.info(
                f"🔍 DETECT[nms] camera={camera_id} "
                f"before={n_pre_nms} after={n_post_nms} "
                f"filtered={filtered_by_nms} "
                f"kept_confs=[{', '.join(f'{d.confidence:.3f}' for d in detections)}]"
            )

        self._save_debug_stage(camera_id, "after_nms", original, detections)

        t_total = time.perf_counter() - t_start
        logger.info(
            f"🔍 DETECT[summary] camera={camera_id} "
            f"raw={n_raw} "
            f"after_adaptive={n_after_adaptive} "
            f"after_nms={len(detections)} "
            f"pipeline_time_ms={t_total*1000:.1f}"
        )

        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement for better detection in warehouse lighting."""
        if not self.use_clahe:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _parse_results(self, results) -> list[Detection]:
        """Extract detections from YOLO results object."""
        detections = []
        if results.boxes is None:
            return detections
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append(Detection(
                bbox=       (x1, y1, x2, y2),
                confidence= conf,
                class_id=   int(box.cls[0]),
                class_name= "person",
            ))
        return detections

    def _merge_tta(
        self,
        dets_a: list[Detection],
        dets_b: list[Detection],
    ) -> list[Detection]:
        """Merge two detection sets from TTA, keeping higher-confidence boxes."""
        merged = list(dets_a)
        for db in dets_b:
            keep = True
            for da in dets_a:
                if da.iou(db) > 0.7 and db.confidence <= da.confidence:
                    keep = False
                    break
            if keep:
                merged.append(db)
        return merged

    def _adaptive_filter(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
        camera_id: str = "unknown",
    ) -> list[Detection]:
        """
        Adaptive confidence thresholding:
          • Very small boxes (area < 0.5% of frame) → slight threshold boost (conf × 1.05)
          • Medium boxes → standard threshold
          • Large boxes (area > 40% of frame) → likely false positive, require conf × 1.35
          • Aspect-ratio boost: tall boxes (height/width > 1.5) get confidence × 1.05
            because they match the human silhouette more closely.

        NOTE: The 0.5% threshold is deliberately low. In a 640×360 frame,
        0.5% = 1,152 pixels ≈ 34×34 px. Very distant people can be this
        small while still being valid detections.
        """
        frame_h, frame_w = frame_shape
        frame_area = frame_h * frame_w
        filtered = []
        for d in detections:
            area_ratio = d.area / frame_area
            aspect = d.height / max(d.width, 1)

            if area_ratio < 0.005:
                # Very small boxes: tiny threshold boost
                min_conf = self.conf * 1.05
                size_class = "tiny"
            elif area_ratio < 0.02:
                # Small boxes: no boost
                min_conf = self.conf
                size_class = "small"
            elif area_ratio > 0.40:
                # Large boxes (likely false positive)
                min_conf = self.conf * 1.35
                size_class = "large"
            else:
                min_conf = self.conf
                size_class = "normal"

            aspect_boost = 1.05 if aspect > 1.5 else 1.0
            effective_conf = d.confidence * aspect_boost
            passed = effective_conf >= min_conf

            logger.debug(
                f"[{camera_id}]   Adaptive: box=({d.x1},{d.y1},{d.x2},{d.y2}) "
                f"conf={d.confidence:.4f} area_ratio={area_ratio:.6f} "
                f"aspect={aspect:.2f} size={size_class} "
                f"min_conf={min_conf:.4f} boost={aspect_boost:.2f} "
                f"eff_conf={effective_conf:.4f} -> {'KEEP' if passed else 'FILTER'}"
            )

            if passed:
                filtered.append(d)

        n_filtered = len(detections) - len(filtered)
        if n_filtered > 0:
            filtered_details = '; '.join(
                f'idx#{i} conf={d.confidence:.3f} area_ratio={d.area/frame_area:.6f} ar={d.height/max(d.width,1):.2f}'
                for i, d in enumerate(detections)
                if not any(f.x1==d.x1 and f.y1==d.y1 for f in filtered)
            )
            logger.info(
                f"🔍 ADAPTIVE_FILTER camera={camera_id} "
                f"total={len(detections)} kept={len(filtered)} filtered={n_filtered} "
                f"filtered_details=[{filtered_details}]"
            )

        return filtered

    def _soft_nms(
        self,
        detections: list[Detection],
        iou_threshold: float = 0.50,
        sigma: float = 0.5,
        score_threshold: float = 0.15,
    ) -> list[Detection]:
        """
        Soft-NMS: decay confidence of overlapping boxes instead of hard removal.
        Preserves detections that are near each other (e.g. people standing close)
        while suppressing redundant boxes.
        """
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        for i in range(len(detections)):
            if detections[i].confidence < score_threshold:
                continue
            for j in range(i + 1, len(detections)):
                if detections[j].confidence < score_threshold:
                    continue
                iou = detections[i].iou(detections[j])
                if iou > iou_threshold:
                    # Gaussian penalty on the lower-confidence box
                    penalty = 1.0 - iou if iou < 0.9 else 0.0
                    detections[j].confidence *= penalty

        return [d for d in detections if d.confidence >= self.conf]

    def detect_with_classes(
        self,
        frame: np.ndarray,
        class_ids: Optional[set[int]] = None,
        camera_id: str = "unknown",
    ) -> list[Detection]:
        """
        Run YOLOv8 inference detecting specified COCO classes.

        Args:
            frame: BGR numpy array (H × W × 3)
            class_ids: Set of COCO class IDs to detect. If None, detects all classes.
            camera_id: Camera identifier for per-camera logging.

        Returns:
            List of Detection objects for the requested classes.
        """
        class_list = list(class_ids) if class_ids is not None else None
        processed = self._preprocess(frame)

        results = self.model(
            processed,
            conf=    self.conf * 0.8,
            iou=     self.iou_thresh,
            classes= class_list,
            imgsz=   self.img_size,
            verbose= False,
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append(Detection(
                bbox=       (x1, y1, x2, y2),
                confidence= conf,
                class_id=   cls_id,
                class_name= self.COCO_NAMES.get(cls_id, f"class_{cls_id}"),
            ))

        logger.debug(
            f"[{camera_id}]   detect_with_classes({len(class_ids or [])} classes): "
            f"{len(detections)} raw, "
            f"classes={[d.class_name for d in detections]}"
        )

        if len(detections) > 1:
            detections = self._soft_nms(detections, iou_threshold=self.iou_thresh)
            logger.debug(
                f"[{camera_id}]   After NMS: {len(detections)} carryable objects"
            )

        return detections

    def detect_carryable_objects(
        self,
        frame: np.ndarray,
        camera_id: str = "unknown",
    ) -> list[Detection]:
        """
        Convenience method: detect objects that could be carried by a person.

        Returns list of Detection for CARRYABLE_CLASSES.
        """
        return self.detect_with_classes(
            frame, class_ids=self.CARRYABLE_CLASSES, camera_id=camera_id,
        )

    @staticmethod
    def _nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
        """Remove overlapping bounding boxes (Non-Max Suppression)."""
        if not detections:
            return []
        detections.sort(key=lambda d: d.confidence, reverse=True)
        kept = []
        for det in detections:
            if all(det.iou(k) < iou_threshold for k in kept):
                kept.append(det)
        return kept


# ── Backend 2: HOG + SVM (OpenCV built-in) ───────────────────────────────────

class HOGDetector(BaseDetector):
    """
    HOG + SVM person detector built into OpenCV.

    No model file needed — works out of the box.
    Slower and less accurate than YOLO, but:
      • Zero dependencies beyond OpenCV
      • Works on any CPU
      • Good enough for prototype / CI

    Used automatically when yolov8n.pt is not available.
    """

    def __init__(
        self,
        confidence_thresh: float = 0.3,
        win_stride:        tuple = (8, 8),
        padding:           tuple = (4, 4),
        scale:             float = 1.05,
    ):
        self.conf       = confidence_thresh
        self.win_stride = win_stride
        self.padding    = padding
        self.scale      = scale

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info("HOGDetector ready (fallback mode — install yolov8n.pt for better accuracy)")

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str = "unknown",
    ) -> list[Detection]:
        """
        Run HOG detection on a single BGR frame.

        HOG works best on frames scaled to ~480px wide.
        We resize internally to speed up detection.
        """
        h, w = frame.shape[:2]

        # Resize to 480 wide for speed (maintains aspect ratio)
        scale_factor = 480 / w
        small = cv2.resize(frame, (480, int(h * scale_factor)))

        # Convert to grayscale (HOG works on intensity, not colour)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        rects, weights = self._hog.detectMultiScale(
            gray,
            winStride= self.win_stride,
            padding=   self.padding,
            scale=     self.scale,
        )

        detections = []
        for (x, y, bw, bh), w_score in zip(rects, weights):
            if float(w_score) < self.conf:
                continue
            # Scale coords back to original frame size
            sx = 1.0 / scale_factor
            x1 = int(x * sx)
            y1 = int(y * sx)
            x2 = int((x + bw) * sx)
            y2 = int((y + bh) * sx)
            detections.append(Detection(
                bbox=       (x1, y1, x2, y2),
                confidence= float(w_score),
                class_id=   0,
                class_name= "person",
            ))

        # Non-max suppression to remove overlapping boxes
        return self._nms(detections, iou_threshold=0.45)

    @staticmethod
    def _nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
        """Remove overlapping bounding boxes (Non-Max Suppression)."""
        if not detections:
            return []
        detections.sort(key=lambda d: d.confidence, reverse=True)
        kept = []
        for det in detections:
            if all(det.iou(k) < iou_threshold for k in kept):
                kept.append(det)
        return kept


# ── Backend 3: Mock (for testing) ─────────────────────────────────────────────

class MockDetector(BaseDetector):
    """
    Returns synthetic detections.
    Used in unit tests and CI pipelines where no model is available.
    """
    import random as _rand

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str = "unknown",
    ) -> list[Detection]:
        import random
        h, w = frame.shape[:2]
        n_persons = random.randint(0, 3)
        dets = []
        for _ in range(n_persons):
            x1 = random.randint(0, w - 80)
            y1 = random.randint(0, h - 150)
            x2 = x1 + random.randint(50, 100)
            y2 = y1 + random.randint(120, 180)
            dets.append(Detection(
                bbox=       (min(x1, w-1), min(y1, h-1), min(x2, w-1), min(y2, h-1)),
                confidence= round(random.uniform(0.5, 0.95), 2),
            ))
        return dets


# ── Factory: auto-pick the best available detector ────────────────────────────

class PersonDetector:
    """
    Public API for person detection.

    Automatically selects the best available backend:
      1. YOLOv8  if yolov8n.pt exists in models/
      2. HOG     as fallback (always available)
      3. Mock    if DETECTOR_MODE=mock in env

    Usage:
        detector = PersonDetector()
        detections = detector.detect(frame)  # list[Detection]
    """

    def __init__(
        self,
        model_path:        str   = "models/yolov8n.pt",
        confidence_thresh: float = 0.25,
        device:            str   = "cpu",
        force_backend:     Optional[str] = None,
        img_size:          Optional[int] = None,
        use_tta:           Optional[bool] = None,
        use_clahe:         Optional[bool] = None,
        debug_dir:         Optional[str] = None,
    ):
        backend_name = force_backend or os.environ.get("DETECTOR_BACKEND", "auto")
        self._backend = self._pick_backend(
            backend_name, model_path, confidence_thresh, device,
            img_size, use_tta, use_clahe, debug_dir,
        )

    @staticmethod
    def _pick_backend(
        mode:       str,
        model_path: str,
        conf:       float,
        device:     str,
        img_size:   Optional[int] = None,
        use_tta:    Optional[bool] = None,
        use_clahe:  Optional[bool] = None,
        debug_dir:  Optional[str] = None,
    ) -> BaseDetector:
        if mode == "mock":
            logger.info("Detector: mock mode")
            return MockDetector()

        if mode == "yolo" or (mode == "auto" and os.path.exists(model_path)):
            try:
                det = YOLODetector(
                    model_path, conf, device,
                    img_size=img_size or 960,
                    use_tta=use_tta or False,
                    use_clahe=use_clahe if use_clahe is not None else True,
                    debug_dir=debug_dir,
                )
                det.warmup()
                return det
            except Exception as e:
                logger.warning(f"YOLO init failed ({e}), falling back to HOG")

        logger.info("Detector: HOG fallback mode")
        return HOGDetector(conf)

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str = "unknown",
    ) -> list[Detection]:
        """Run person detection. Returns empty list on error (never raises)."""
        try:
            return self._backend.detect(frame, camera_id)
        except Exception as e:
            logger.error(f"[{camera_id}] Detection error: {e}")
            return []

    def detect_with_classes(
        self,
        frame: np.ndarray,
        class_ids: Optional[set[int]] = None,
        camera_id: str = "unknown",
    ) -> list[Detection]:
        """Run detection for specified COCO classes. Returns empty list on error."""
        if not hasattr(self._backend, "detect_with_classes"):
            return []
        try:
            return self._backend.detect_with_classes(frame, class_ids, camera_id)
        except Exception as e:
            logger.error(f"[{camera_id}] Multi-class detection error: {e}")
            return []

    def detect_carryable_objects(
        self,
        frame: np.ndarray,
        camera_id: str = "unknown",
    ) -> list[Detection]:
        """Detect objects that could be carried by a person. Returns empty list on error."""
        if not hasattr(self._backend, "detect_carryable_objects"):
            return []
        try:
            return self._backend.detect_carryable_objects(frame, camera_id)
        except Exception as e:
            logger.error(f"[{camera_id}] Carryable object detection error: {e}")
            return []

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__
