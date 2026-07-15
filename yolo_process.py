
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from ultralytics import YOLO


Box = Tuple[int, int, int, int]


class YOLOprocess:


    image_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    video_formats = (".mp4", ".avi", ".mov")

    def __init__(self, model_path = "models/best.pt", confidence: float = 0.6):
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def process_image(self, image_path):

        result = self.model.predict(str(image_path), conf=self.confidence, verbose=False)[0]
        if result.boxes is None:
            return []

        height, width = result.orig_shape
        boxes = []
        for x1, y1, x2, y2 in result.boxes.xyxy.cpu().tolist():
            left = max(0, min(width, int(x1)))
            top = max(0, min(height, int(y1)))
            right = max(0, min(width, int(x2)))
            bottom = max(0, min(height, int(y2)))
            if right - left >= 8 and bottom - top >= 8:
                boxes.append((left, top, right, bottom))
        return sorted(boxes, key=lambda box: (box[0] + box[2]) / 2)

    def process_frame(self, frame) :

        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        if result.masks is None:
            return []
        polygons = []
        for polygon in result.masks.xy:
            points = np.rint(polygon).astype(np.int32)
            if len(points) >= 3:
                polygons.append(points)
        return polygons

    def __call__(self, file_path):
        path = Path(file_path)
        if path.suffix.lower() in self.image_formats:
            return self.process_image(path)
        raise ValueError("处理失败，请检查文件类型")
