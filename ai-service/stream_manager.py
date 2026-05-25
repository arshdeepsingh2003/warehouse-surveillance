from pipeline.frame_extractor import FrameExtractor
from pipeline.person_tracker import PersonTracker
from pipeline.activity_detector import ActivityDetector
from pipeline.anomaly_classifier import AnomalyClassifier

class StreamManager:
    def __init__(self):
        self.extractors: dict[str, FrameExtractor] = {}
        self.trackers: dict[str, PersonTracker] = {}

    def start_camera(self, camera_id: str, rtsp_url: str):
        ...

    def stop_camera(self, camera_id: str):
        ...
