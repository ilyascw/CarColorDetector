"""Console entry points for the API and interactive demo."""

from __future__ import annotations

import os


def run_api() -> None:
    import uvicorn

    uvicorn.run(
        "car_color_detector.api:app",
        host=os.getenv("CAR_COLOR_HOST", "0.0.0.0"),
        port=int(os.getenv("CAR_COLOR_PORT", "8000")),
    )


def run_web() -> None:
    from car_color_detector.web import create_demo

    create_demo().launch(
        server_name=os.getenv("CAR_COLOR_HOST", "0.0.0.0"),
        server_port=int(os.getenv("CAR_COLOR_PORT", "7860")),
    )
