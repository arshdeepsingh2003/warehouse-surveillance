import cv2

class FrameExtractor:
    def __init__(self, rtsp_url: str):
        self.cap = cv2.VideoCapture(rtsp_url)

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
