import gradio as gr
import cv2
import numpy as np
from typing import Tuple, List
from CarColorProcessor import CarColorProcessor  # Импортируем твой класс

# Инициализация обработчика
processor = CarColorProcessor(
    yolo_model_path="models/yolov8n.pt",  
    sam_checkpoint="models/sam_vit_b_01ec64.pth",  
    sam_model_type="vit_b",
    color_database='data/colors.csv',
    max_size=512,
    device="cpu",
)

def process_image(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Обработка изображения через CarColorProcessor для Gradio.

    Args:
        image: Входное изображение (RGB, numpy, как даёт Gradio).

    Returns:
        Кортеж (выходное изображение, текстовое описание).
    """
    # Gradio даёт RGB, конвертируем в BGR для OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        # Обработка
        result = processor.process(image_bgr)
        
        # Конвертируем выходное изображение обратно в RGB для Gradio
        output_image = cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB)
        
        # Формируем текст описания
        description = "\n".join(result["description"])
        
        return output_image, description
    
    except Exception as e:
        return image, f"Ошибка обработки: {str(e)}"

# Создание Gradio-интерфейса
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Загрузите изображение автомобиля (jpg/png)"),
    outputs=[
        gr.Image(type="numpy", label="Результат (bbox и цвета)"),
        gr.Textbox(label="Описание цветов")
    ],
    title="Детекция цветов автомобилей",
    description="Загрузите фото автомобиля, чтобы определить его цвета.",
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)