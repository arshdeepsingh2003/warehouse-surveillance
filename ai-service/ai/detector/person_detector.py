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

    def detect(self, frame: np.ndarray) -> list[Detection]:
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

    For warehouse surveillance on CPU: yolov8n.pt is the right choice.
    """

    def __init__(
        self,
        model_path:       str   = "models/yolov8n.pt",
        confidence_thresh: float = 0.25,   # lowered from 0.40 — yolov8n needs lower threshold for recall
        device:           str   = "cpu",   # "cuda:0" for GPU
        img_size:         int   = 640,
    ):
        from ultralytics import YOLO   # import here so HOG still works without it

        self.conf       = confidence_thresh
        self.iou_thresh = 0.45   # NMS threshold — stricter than YOLO default (0.7)
        self.img_size   = img_size
        self.device     = device

        logger.info(f"Loading YOLO model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)

        # Class index for "person" in COCO dataset
        self._person_class = 0
        logger.info("YOLODetector ready")

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

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLOv8 inference on a single BGR frame.

        Applies YOLO's internal NMS + an explicit secondary NMS pass to
        suppress overlapping detections that can cause ghost boxes.

        Args:
            frame: BGR numpy array (H × W × 3)

        Returns:
            List of Detection objects, persons only, above confidence threshold.
        """
        results = self.model(
            frame,
            conf=    self.conf,
            iou=     self.iou_thresh,
            classes= [self._person_class],   # only detect persons
            imgsz=   self.img_size,
            verbose= False,
        )[0]

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

        # Secondary NMS — catch any overlapping boxes YOLO's internal NMS missed
        if len(detections) > 1:
            detections = self._nms(detections, iou_threshold=self.iou_thresh)

        return detections

    def detect_with_classes(
        self,
        frame: np.ndarray,
        class_ids: Optional[set[int]] = None,
    ) -> list[Detection]:
        """
        Run YOLOv8 inference detecting specified COCO classes.

        Args:
            frame: BGR numpy array (H × W × 3)
            class_ids: Set of COCO class IDs to detect. If None, detects all classes.

        Returns:
            List of Detection objects for the requested classes.
        """
        class_list = list(class_ids) if class_ids is not None else None

        results = self.model(
            frame,
            conf=    self.conf,
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

        if len(detections) > 1:
            detections = self._nms(detections, iou_threshold=self.iou_thresh)

        return detections

    def detect_carryable_objects(self, frame: np.ndarray) -> list[Detection]:
        """
        Convenience method: detect objects that could be carried by a person.

        Returns list of Detection for CARRYABLE_CLASSES.
        """
        return self.detect_with_classes(frame, class_ids=self.CARRYABLE_CLASSES)

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

    def detect(self, frame: np.ndarray) -> list[Detection]:
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

    def detect(self, frame: np.ndarray) -> list[Detection]:
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
        force_backend:     Optional[str] = None,   # "yolo" | "hog" | "mock"
    ):
        backend_name = force_backend or os.environ.get("DETECTOR_BACKEND", "auto")
        self._backend = self._pick_backend(backend_name, model_path, confidence_thresh, device)

    @staticmethod
    def _pick_backend(
        mode:       str,
        model_path: str,
        conf:       float,
        device:     str,
    ) -> BaseDetector:
        if mode == "mock":
            logger.info("Detector: mock mode")
            return MockDetector()

        if mode == "yolo" or (mode == "auto" and os.path.exists(model_path)):
            try:
                det = YOLODetector(model_path, conf, device)
                det.warmup()
                return det
            except Exception as e:
                logger.warning(f"YOLO init failed ({e}), falling back to HOG")

        logger.info("Detector: HOG fallback mode")
        return HOGDetector(conf)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run person detection. Returns empty list on error (never raises)."""
        try:
            return self._backend.detect(frame)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def detect_with_classes(
        self,
        frame: np.ndarray,
        class_ids: Optional[set[int]] = None,
    ) -> list[Detection]:
        """Run detection for specified COCO classes. Returns empty list on error."""
        if not hasattr(self._backend, "detect_with_classes"):
            return []
        try:
            return self._backend.detect_with_classes(frame, class_ids)
        except Exception as e:
            logger.error(f"Multi-class detection error: {e}")
            return []

    def detect_carryable_objects(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects that could be carried by a person. Returns empty list on error."""
        if not hasattr(self._backend, "detect_carryable_objects"):
            return []
        try:
            return self._backend.detect_carryable_objects(frame)
        except Exception as e:
            logger.error(f"Carryable object detection error: {e}")
            return []

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__
