import cv2
import numpy as np
from fastapi.testclient import TestClient

from car_color_detector.api import create_app
from car_color_detector.catalog import ColorCatalog, ColorEntry
from car_color_detector.models import BoundingBox
from car_color_detector.pipeline import CarColorProcessor
from tests.test_pipeline import FixedDetector, FullMaskSegmenter


def make_processor() -> CarColorProcessor:
    return CarColorProcessor(
        detector=FixedDetector({"vehicle_0": BoundingBox(1, 1, 18, 18)}),
        segmenter=FullMaskSegmenter(),
        catalog=ColorCatalog([ColorEntry("red", "Красный", (200, 0, 0))]),
    )


def encode_test_image() -> bytes:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 200)
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded
    return buffer.tobytes()


def test_health_reports_loaded_model() -> None:
    with TestClient(create_app(make_processor)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_serializable_colors() -> None:
    with TestClient(create_app(make_processor)) as client:
        response = client.post(
            "/predict",
            files={"file": ("car.png", encode_test_image(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cars"]["vehicle_0"]["color_names"] == ["Красный"]
    assert payload["image"].startswith("data:image/jpeg;base64,")


def test_predict_rejects_unsupported_media_type() -> None:
    with TestClient(create_app(make_processor)) as client:
        response = client.post(
            "/predict",
            files={"file": ("car.gif", b"not-an-image", "image/gif")},
        )

    assert response.status_code == 415
