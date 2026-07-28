"""Gradio interface with lazy model initialization."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock
from typing import Any, cast

import cv2
import gradio as gr
import numpy as np

from car_color_detector.factory import build_processor
from car_color_detector.models import ImageArray
from car_color_detector.pipeline import CarColorProcessor

LOGGER = logging.getLogger(__name__)


class LazyProcessor:
    def __init__(self, factory: Callable[[], CarColorProcessor]) -> None:
        self._factory = factory
        self._processor: CarColorProcessor | None = None
        self._lock = Lock()

    def get(self) -> CarColorProcessor:
        if self._processor is None:
            with self._lock:
                if self._processor is None:
                    self._processor = self._factory()
        return self._processor


def create_demo(
    processor_factory: Callable[[], CarColorProcessor] = build_processor,
) -> Any:
    lazy_processor = LazyProcessor(processor_factory)

    def process_image(image_rgb: ImageArray | None) -> tuple[ImageArray | None, str]:
        if image_rgb is None:
            return None, "Загрузите изображение автомобиля."
        try:
            image_bgr = cast(
                ImageArray,
                cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR),
            )
            result = lazy_processor.get().process(image_bgr)
            output_rgb = cast(
                ImageArray,
                cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB),
            )
            return output_rgb, "\n".join(result.description)
        except FileNotFoundError as error:
            return image_rgb, str(error)
        except Exception:
            LOGGER.exception("Gradio inference failed")
            return image_rgb, "Не удалось обработать изображение. Проверьте логи приложения."

    with gr.Blocks(title="Car Color Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Car Color Detector

            Детекция автомобилей, сегментация кузова и оценка доминирующих цветов.
            Результат содержит bounding box, цветовую палитру и названия оттенков.
            """
        )
        with gr.Row():
            input_image = gr.Image(type="numpy", label="Исходное изображение")
            output_image = gr.Image(type="numpy", label="Результат")
        run_button = gr.Button("Определить цвета", variant="primary")
        description = gr.Textbox(label="Найденные автомобили и цвета", lines=5)
        run_button.click(
            fn=process_image,
            inputs=input_image,
            outputs=[output_image, description],
        )
    return demo
