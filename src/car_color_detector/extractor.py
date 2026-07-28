"""Dominant-color extraction from a segmented vehicle region."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from car_color_detector.models import BoundingBox, ImageArray, MaskArray, RgbColor


class DominantColorExtractor:
    """Extract color clusters after removing near-black and overexposed pixels."""

    def __init__(
        self,
        *,
        clusters: int = 2,
        min_pixels: int = 10,
        random_state: int = 42,
    ) -> None:
        if clusters < 1:
            raise ValueError("clusters must be positive")
        self._clusters = clusters
        self._min_pixels = min_pixels
        self._random_state = random_state

    def extract(
        self,
        image_bgr: ImageArray,
        mask: MaskArray,
        bbox: BoundingBox,
    ) -> tuple[RgbColor, ...]:
        pixels_bgr = image_bgr[mask]
        if pixels_bgr.size == 0:
            pixels_bgr = image_bgr[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max].reshape(-1, 3)
        if pixels_bgr.size == 0:
            return ()

        pixels_rgb = pixels_bgr[:, ::-1]
        normalized = pixels_rgb.astype(np.float64) / 255.0
        value = normalized.max(axis=1)
        valid_pixels = pixels_rgb[(value > 0.1) & (value < 0.95)]
        if len(valid_pixels) < self._min_pixels:
            return ()

        unique_colors = np.unique(valid_pixels, axis=0)
        cluster_count = min(self._clusters, len(unique_colors))
        model = KMeans(
            n_clusters=cluster_count,
            n_init="auto",
            random_state=self._random_state,
        )
        labels = cast(NDArray[np.int_], model.fit_predict(valid_pixels))
        centers = cast(NDArray[np.float64], model.cluster_centers_)
        counts = np.bincount(labels, minlength=cluster_count)
        ordered = centers[np.argsort(counts)[::-1]].round().astype(np.uint8)
        return tuple((int(row[0]), int(row[1]), int(row[2])) for row in ordered)
