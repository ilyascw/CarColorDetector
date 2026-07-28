"""Adapters for YOLOv8 vehicle detection and Segment Anything segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from car_color_detector.models import BoundingBox, ImageArray, MaskArray


class YoloVehicleDetector:
    """Ultralytics adapter that keeps only COCO vehicle classes."""

    def __init__(self, model_path: str, *, confidence: float = 0.25) -> None:
        from ultralytics import YOLO

        self._model: Any = YOLO(model_path)
        self._confidence = confidence

    def detect(self, image_bgr: ImageArray) -> Mapping[str, BoundingBox]:
        result: Any = self._model(
            image_bgr,
            conf=self._confidence,
            agnostic_nms=True,
            verbose=False,
        )[0]
        detections: dict[str, BoundingBox] = {}
        car_index = 0
        for box in result.boxes:
            class_id = int(box.cls.item())
            if str(self._model.names[class_id]) not in {"car", "truck", "bus"}:
                continue
            coordinates = cast(
                NDArray[np.int_],
                box.xyxy[0].detach().cpu().numpy().round().astype(np.int_),
            )
            detections[f"vehicle_{car_index}"] = BoundingBox(
                x_min=int(coordinates[0]),
                y_min=int(coordinates[1]),
                x_max=int(coordinates[2]),
                y_max=int(coordinates[3]),
            )
            car_index += 1
        return detections


class SamVehicleSegmenter:
    """Segment Anything adapter with a bounding-box fallback handled upstream."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        model_type: str = "vit_b",
        device: str = "cpu",
        minimum_score: float = 0.5,
        bbox_margin: float = 0.1,
    ) -> None:
        from segment_anything import SamPredictor, sam_model_registry

        if model_type not in sam_model_registry:
            raise ValueError(f"Unsupported SAM model type: {model_type}")
        model: Any = sam_model_registry[model_type](checkpoint=checkpoint_path)
        model.to(device=device)
        self._predictor: Any = SamPredictor(model)
        self._minimum_score = minimum_score
        self._bbox_margin = bbox_margin

    def segment(
        self,
        image_bgr: ImageArray,
        detections: Mapping[str, BoundingBox],
    ) -> Mapping[str, MaskArray]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(image_rgb)
        masks_by_car: dict[str, MaskArray] = {}
        height, width = image_bgr.shape[:2]

        for car_id, bbox in detections.items():
            expanded = bbox.expand(self._bbox_margin, width=width, height=height)
            box = np.asarray(expanded.as_list(), dtype=np.float32)
            masks, scores, _ = self._predictor.predict(
                box=box,
                multimask_output=False,
            )
            score_array = np.asarray(scores, dtype=np.float64)
            if len(masks) == 0 or score_array.size == 0:
                continue
            best_index = int(score_array.argmax())
            if float(score_array[best_index]) < self._minimum_score:
                continue
            masks_by_car[car_id] = np.asarray(masks[best_index], dtype=np.bool_)
        return masks_by_car
