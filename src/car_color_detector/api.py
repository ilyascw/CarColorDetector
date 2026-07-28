"""FastAPI delivery layer for the vehicle color pipeline."""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import cast

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from car_color_detector.factory import build_processor
from car_color_detector.models import CarPrediction, ImageArray, ProcessingResult
from car_color_detector.pipeline import CarColorProcessor

LOGGER = logging.getLogger(__name__)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class CarResponse(BaseModel):
    bbox: list[int]
    colors_rgb: list[list[int]]
    color_names: list[str]


class PredictionResponse(BaseModel):
    image: str
    cars: dict[str, CarResponse]
    description: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def _serialize_car(prediction: CarPrediction) -> CarResponse:
    return CarResponse(
        bbox=prediction.bbox.as_list(),
        colors_rgb=[list(color) for color in prediction.colors_rgb],
        color_names=list(prediction.color_names),
    )


def _encode_image(image: ImageArray) -> str:
    encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not encoded:
        raise ValueError("Failed to encode inference result")
    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def create_app(
    processor_factory: Callable[[], CarColorProcessor] = build_processor,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.processor = await run_in_threadpool(processor_factory)
        yield

    application = FastAPI(
        title="Car Color Detector API",
        description="Vehicle detection, segmentation and dominant body-color estimation",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        status_code=status.HTTP_200_OK,
    )
    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Supported formats: JPEG, PNG and WebP",
            )

        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Image exceeds the 10 MiB upload limit",
            )

        buffer = np.frombuffer(contents, dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded file is not a valid image",
            )
        image = cast(ImageArray, decoded)

        try:
            processor = cast(CarColorProcessor, request.app.state.processor)
            result: ProcessingResult = await run_in_threadpool(processor.process, image)
            return PredictionResponse(
                image=_encode_image(result.image),
                cars={
                    car_id: _serialize_car(prediction) for car_id, prediction in result.cars.items()
                },
                description=list(result.description),
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        except Exception as error:
            LOGGER.exception("Vehicle color inference failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference failed",
            ) from error

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=hasattr(request.app.state, "processor"),
        )

    return application


app = create_app()
