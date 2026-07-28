"""Environment-driven composition root for production inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

from car_color_detector.catalog import ColorCatalog
from car_color_detector.inference import SamVehicleSegmenter, YoloVehicleDetector
from car_color_detector.pipeline import CarColorProcessor


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    yolo_model_path: Path
    sam_checkpoint_path: Path
    sam_model_type: str = "vit_b"
    device: str = "cpu"
    max_image_size: int = 1024

    @classmethod
    def from_environment(cls) -> InferenceSettings:
        return cls(
            yolo_model_path=Path(
                os.getenv("CAR_COLOR_YOLO_MODEL", "models/yolov8n.pt")
            ).expanduser(),
            sam_checkpoint_path=Path(
                os.getenv("CAR_COLOR_SAM_CHECKPOINT", "models/sam_vit_b_01ec64.pth")
            ).expanduser(),
            sam_model_type=os.getenv("CAR_COLOR_SAM_MODEL_TYPE", "vit_b"),
            device=os.getenv("CAR_COLOR_DEVICE", "cpu"),
            max_image_size=int(os.getenv("CAR_COLOR_MAX_IMAGE_SIZE", "1024")),
        )

    def validate(self) -> None:
        missing = [
            str(path)
            for path in (self.yolo_model_path, self.sam_checkpoint_path)
            if not path.is_file()
        ]
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(
                f"Model files are missing: {joined}. See models/README.md for download commands."
            )


def build_processor(settings: InferenceSettings | None = None) -> CarColorProcessor:
    resolved = settings or InferenceSettings.from_environment()
    resolved.validate()

    catalog_resource = files("car_color_detector").joinpath("data/colors.csv")
    with as_file(catalog_resource) as catalog_path:
        catalog = ColorCatalog.from_csv(catalog_path)

    return CarColorProcessor(
        detector=YoloVehicleDetector(str(resolved.yolo_model_path)),
        segmenter=SamVehicleSegmenter(
            str(resolved.sam_checkpoint_path),
            model_type=resolved.sam_model_type,
            device=resolved.device,
        ),
        catalog=catalog,
        max_size=resolved.max_image_size,
    )
