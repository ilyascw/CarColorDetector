from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import Dict, Any, List
from CarColorProcessor import CarColorProcessor  
import base64

app = FastAPI(
    title="Car Color Detection API",
    description="API для определения цветов автомобилей на изображении",
    version="1.0.0",
)

# Инициализация обработчика
processor = CarColorProcessor(
    yolo_model_path="models/yolov8n.pt",  
    sam_checkpoint="models/sam_vit_b_01ec64.pth",  
    sam_model_type="vit_b",
    color_database='data/colors.csv',
    max_size=512,
    device="cpu",
)

@app.post("/predict", response_model=Dict[str, Any])
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Эндпоинт для обработки изображения и определения цветов автомобилей.

    Args:
        file: Загруженное изображение (jpg, png).

    Returns:
        JSON с результатами: изображение (base64), словарь cars, описание.

    Raises:
        HTTPException: Если файл некорректен или обработка не удалась.
    """
    # Проверка формата файла
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Поддерживаются только JPEG и PNG")

    try:
        # Чтение изображения
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Не удалось декодировать изображение")

        # Обработка
        result = processor.process(image)

        # Кодирование выходного изображения в base64
        _, buffer = cv2.imencode(".jpg", result["image"])
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # Формирование ответа
        response = {
            "image": f"data:image/jpeg;base64,{image_base64}",
            "cars": {
                car_id: {
                    "bbox": data["bbox"].tolist(),
                    "colors_rgb": data["colors_rgb"].tolist() if data["colors_rgb"].size else [],
                    "color_names": data["color_names"]
                }
                for car_id, data in result["cars"].items()
            },
            "description": result["description"],
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Проверка состояния API.

    Returns:
        JSON с сообщением о статусе.
    """
    return {"status": "healthy"}
