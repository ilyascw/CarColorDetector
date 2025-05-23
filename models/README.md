# Инструкции по загрузке моделей

Папка `models/` должна содержать две модели:
1. **yolov8n.pt** (~6 МБ): YOLOv8 Nano для детекции автомобилей.
2. **sam_vit_h.pth** (~375 МБ): Segment Anything Model (SAM) для сегментации.

## Как загрузить
1. **yolov8n.pt**:
   - Скачайте с официального репозитория Ultralytics:
     ```bash
     wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -O models/yolov8n.pt
     ```
   - Или установите `ultralytics` и используйте API:
     ```python
     from ultralytics import YOLO
     model = YOLO("yolov8n.pt")  # Автоматически загрузит в ~/.cache/ultralytics
     ```
     Затем переместите в `models/yolov8n.pt`.

2. **sam_vit_h.pth**:
   - Скачайте с официального репозитория SAM:
     ```bash
     wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h.pth
     ```
   - Проверьте целостность (размер ~375 МБ):
     ```bash
     ls -lh models/sam_vit_h.pth
     ```

## Примечания
- Убедитесь, что пути в `CarColorProcessor.py` или `app.py` указывают на `models/yolov8n.pt` и `models/sam_vit_h.pth`.
- Если модели не загружены, запуск `app.py` или `API.py` выдаст ошибку `FileNotFoundError`.