"""Typed domain models shared by the processing and delivery layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]
MaskArray: TypeAlias = NDArray[np.bool_]
RgbColor: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Inclusive-exclusive image coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    def clamp(self, *, width: int, height: int) -> BoundingBox:
        x_min = min(max(self.x_min, 0), width)
        y_min = min(max(self.y_min, 0), height)
        x_max = min(max(self.x_max, x_min), width)
        y_max = min(max(self.y_max, y_min), height)
        return BoundingBox(x_min, y_min, x_max, y_max)

    def expand(self, margin: float, *, width: int, height: int) -> BoundingBox:
        dx = round(self.width * margin)
        dy = round(self.height * margin)
        return BoundingBox(
            self.x_min - dx,
            self.y_min - dy,
            self.x_max + dx,
            self.y_max + dy,
        ).clamp(width=width, height=height)

    def as_list(self) -> list[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass(frozen=True, slots=True)
class CarPrediction:
    """Colors inferred for one detected vehicle."""

    bbox: BoundingBox
    colors_rgb: tuple[RgbColor, ...]
    color_names: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "bbox": self.bbox.as_list(),
            "colors_rgb": [list(color) for color in self.colors_rgb],
            "color_names": list(self.color_names),
        }


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Complete pipeline response before transport serialization."""

    image: ImageArray
    cars: dict[str, CarPrediction]
    description: tuple[str, ...]
