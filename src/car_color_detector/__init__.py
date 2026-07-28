"""Vehicle color detection pipeline."""

from car_color_detector.models import BoundingBox, CarPrediction, ProcessingResult
from car_color_detector.pipeline import CarColorProcessor

__all__ = [
    "BoundingBox",
    "CarColorProcessor",
    "CarPrediction",
    "ProcessingResult",
]

__version__ = "1.0.0"
