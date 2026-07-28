from collections.abc import Mapping

import numpy as np

from car_color_detector.catalog import ColorCatalog, ColorEntry
from car_color_detector.models import BoundingBox, ImageArray, MaskArray
from car_color_detector.pipeline import CarColorProcessor


class FixedDetector:
    def __init__(self, detections: Mapping[str, BoundingBox]) -> None:
        self._detections = detections

    def detect(self, image_bgr: ImageArray) -> Mapping[str, BoundingBox]:
        return self._detections


class FullMaskSegmenter:
    def segment(
        self,
        image_bgr: ImageArray,
        detections: Mapping[str, BoundingBox],
    ) -> Mapping[str, MaskArray]:
        return {car_id: np.ones(image_bgr.shape[:2], dtype=np.bool_) for car_id in detections}


class MissingMaskSegmenter:
    def segment(
        self,
        image_bgr: ImageArray,
        detections: Mapping[str, BoundingBox],
    ) -> Mapping[str, MaskArray]:
        return {}


def build_processor(
    detections: Mapping[str, BoundingBox],
    *,
    missing_mask: bool = False,
) -> CarColorProcessor:
    catalog = ColorCatalog(
        [
            ColorEntry("red", "Красный", (200, 0, 0)),
            ColorEntry("blue", "Синий", (0, 0, 200)),
        ]
    )
    return CarColorProcessor(
        detector=FixedDetector(detections),
        segmenter=MissingMaskSegmenter() if missing_mask else FullMaskSegmenter(),
        catalog=catalog,
        max_size=100,
    )


def test_pipeline_returns_no_car_message() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)

    result = build_processor({}).process(image)

    assert result.cars == {}
    assert result.description == ("Автомобили не найдены",)


def test_pipeline_extracts_color_and_renders_prediction() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 200)
    processor = build_processor({"vehicle_0": BoundingBox(5, 5, 50, 35)})

    result = processor.process(image)

    prediction = result.cars["vehicle_0"]
    assert prediction.color_names == ("Красный",)
    assert prediction.colors_rgb == ((200, 0, 0),)
    assert result.image.shape == image.shape
    assert not np.array_equal(result.image, image)


def test_pipeline_falls_back_to_bbox_when_segmenter_has_no_mask() -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[5:25, 10:30] = (0, 0, 180)
    processor = build_processor(
        {"vehicle_0": BoundingBox(10, 5, 30, 25)},
        missing_mask=True,
    )

    result = processor.process(image)

    assert result.cars["vehicle_0"].color_names == ("Красный",)


def test_pipeline_rejects_non_rgb_image() -> None:
    processor = build_processor({})
    invalid_image = np.zeros((20, 20), dtype=np.uint8)

    try:
        processor.process(invalid_image)
    except ValueError as error:
        assert "three channels" in str(error)
    else:
        raise AssertionError("ValueError was not raised")
