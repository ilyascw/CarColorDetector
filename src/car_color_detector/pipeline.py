"""Application pipeline independent from concrete neural-network adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import cv2
import numpy as np

from car_color_detector.catalog import ColorCatalog
from car_color_detector.extractor import DominantColorExtractor
from car_color_detector.models import (
    BoundingBox,
    CarPrediction,
    ImageArray,
    MaskArray,
    ProcessingResult,
)


class VehicleDetector(Protocol):
    def detect(self, image_bgr: ImageArray) -> Mapping[str, BoundingBox]: ...


class VehicleSegmenter(Protocol):
    def segment(
        self,
        image_bgr: ImageArray,
        detections: Mapping[str, BoundingBox],
    ) -> Mapping[str, MaskArray]: ...


class CarColorProcessor:
    """Orchestrates resize, detection, segmentation, clustering and rendering."""

    def __init__(
        self,
        *,
        detector: VehicleDetector,
        segmenter: VehicleSegmenter,
        catalog: ColorCatalog,
        extractor: DominantColorExtractor | None = None,
        max_size: int = 1024,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self._detector = detector
        self._segmenter = segmenter
        self._catalog = catalog
        self._extractor = extractor or DominantColorExtractor()
        self._max_size = max_size

    def process(self, image_bgr: ImageArray) -> ProcessingResult:
        image = self._preprocess(image_bgr)
        detections = {
            car_id: bbox.clamp(width=image.shape[1], height=image.shape[0])
            for car_id, bbox in self._detector.detect(image).items()
            if bbox.width > 0 and bbox.height > 0
        }
        if not detections:
            return ProcessingResult(
                image=image.copy(),
                cars={},
                description=("Автомобили не найдены",),
            )

        masks = self._segmenter.segment(image, detections)
        predictions: dict[str, CarPrediction] = {}
        for car_id, bbox in detections.items():
            mask = self._validated_mask(masks.get(car_id), bbox, image.shape[:2])
            colors = self._extractor.extract(image, mask, bbox)
            predictions[car_id] = CarPrediction(
                bbox=bbox,
                colors_rgb=colors,
                color_names=tuple(self._catalog.nearest_name(color) for color in colors),
            )

        rendered = render_predictions(image, predictions)
        return ProcessingResult(
            image=rendered,
            cars=predictions,
            description=describe_predictions(predictions),
        )

    def _preprocess(self, image_bgr: ImageArray) -> ImageArray:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0:
            raise ValueError("Expected a non-empty BGR image with three channels")
        if image_bgr.dtype != np.uint8:
            raise ValueError("Expected an image with uint8 pixels")

        height, width = image_bgr.shape[:2]
        scale = self._max_size / max(height, width)
        if scale >= 1:
            return np.ascontiguousarray(image_bgr)
        resized = cv2.resize(
            image_bgr,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(resized, dtype=np.uint8)

    @staticmethod
    def _validated_mask(
        candidate: MaskArray | None,
        bbox: BoundingBox,
        image_shape: tuple[int, int],
    ) -> MaskArray:
        if candidate is not None and candidate.shape == image_shape and candidate.any():
            return candidate.astype(np.bool_, copy=False)
        mask = np.zeros(image_shape, dtype=np.bool_)
        mask[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max] = True
        return mask


def describe_predictions(predictions: Mapping[str, CarPrediction]) -> tuple[str, ...]:
    lines: list[str] = []
    for car_id, prediction in predictions.items():
        if not prediction.colors_rgb:
            lines.append(f"{car_id}: недостаточно валидных пикселей")
            continue
        summary = ", ".join(
            f"{name} — RGB{color}"
            for name, color in zip(
                prediction.color_names,
                prediction.colors_rgb,
                strict=True,
            )
        )
        lines.append(f"{car_id}: {summary}")
    return tuple(lines)


def render_predictions(
    image_bgr: ImageArray,
    predictions: Mapping[str, CarPrediction],
) -> ImageArray:
    output = image_bgr.copy()
    image_height, image_width = output.shape[:2]

    for car_id, prediction in predictions.items():
        bbox = prediction.bbox
        cv2.rectangle(
            output,
            (bbox.x_min, bbox.y_min),
            (bbox.x_max, bbox.y_max),
            (35, 45, 235),
            2,
        )
        label_y = max(14, bbox.y_min - 6)
        cv2.putText(
            output,
            car_id,
            (bbox.x_min, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (35, 45, 235),
            1,
            cv2.LINE_AA,
        )

        if not prediction.colors_rgb:
            continue
        strip_height = min(24, max(8, bbox.height // 8))
        strip_y_min = min(max(0, bbox.y_max - strip_height), image_height - strip_height)
        segment_width = max(1, bbox.width // len(prediction.colors_rgb))
        for index, color_rgb in enumerate(prediction.colors_rgb):
            x_min = min(image_width - 1, bbox.x_min + index * segment_width)
            x_max = min(
                image_width - 1,
                bbox.x_max if index == len(prediction.colors_rgb) - 1 else x_min + segment_width,
            )
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.rectangle(
                output,
                (x_min, strip_y_min),
                (x_max, strip_y_min + strip_height),
                color_bgr,
                -1,
            )
    return output
