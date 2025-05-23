# Детекция цвета автомобиля

Проект компьютерного зрения для определения цвета автомобиля с использованием YOLOv8 для детекции объектов, Segment Anything Model (SAM) для сегментации и кластеризации K-means для извлечения цветов. Включает веб-интерфейс Gradio для интерактивной демонстрации и FastAPI для программного доступа.

## Возможности
- Детекция автомобилей на изображениях с помощью YOLOv8 Nano (`yolov8n.pt`).
- Сегментация областей автомобиля с помощью SAM (`sam_vit_h.pth`).
- Извлечение основных цветов через K-means и сопоставление с базой цветов (`data/colors.csv`).
- Интерактивный интерфейс Gradio (`app.py`) для загрузки изображений и просмотра результатов (рамки, полоски цветов, описание).
- FastAPI-эндпоинт (`API.py`) для интеграции с другими приложениями.
- Поддержка Docker для простого развёртывания.
- Jupyter Notebook (`notebooks/ste_by_step_processing.ipynb`) с пошаговым объяснением пайплайна.

## Требования
- Python 3.10+
- Docker (опционально, для контейнерного запуска)
- ~4 ГБ оперативной памяти и ~3 ГБ на диске
- Интернет для загрузки моделей (`yolov8n.pt`, `sam_vit_h.pth`)

## Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/yourusername/car-color-detection.git
cd car-color-detection
```

### 2. Создание виртуального окружения
```bash
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
```

### 3. Установка зависимостей
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Загрузка моделей
Следуйте инструкциям в `models/README.md` для загрузки:
- `yolov8n.pt` (~6 МБ)
- `sam_vit_h.pth` (~375 МБ)

Поместите их в папку `models/`.

## Использование

### Вариант 1: Веб-интерфейс Gradio
Запустите приложение Gradio для интерактивной демонстрации:
```bash
python app.py
```
- Откройте `http://localhost:7860` в браузере.
- Загрузите изображение автомобиля, чтобы увидеть рамки, полоски цветов и описание.
- Время обработки: ~5–6 секунд на изображение.

### Вариант 2: FastAPI-эндпоинт
Запустите сервер FastAPI для программного доступа:
```bash
uvicorn API:app --host 0.0.0.0 --port 8000
```
- Откройте `http://localhost:8000/docs` для Swagger UI.
- Пример POST-запроса к `/predict`:
  ```bash
  curl -X POST -F "image=@car.jpg" http://localhost:8000/predict
  ```
- Возвращает JSON с обнаруженными цветами и рамками.

### Вариант 3: Docker
Соберите и запустите проект в Docker-контейнере:
```bash
docker build -t car-color-demo .
docker run -p 7860:7860 car-color-demo
```
- Откройте `http://localhost:7860` для Gradio.
- Для остановки контейнера:
  ```bash
  docker ps  # Найдите CONTAINER ID
  docker stop <CONTAINER_ID>
  ```

### Вариант 4: Jupyter Notebook
Изучите пайплайн пошагово:
```bash
pip install jupyter
jupyter notebook notebooks/ste_by_step_processing.ipynb
```

## Структура проекта
- `CarColorProcessor.py`: Основная логика детекции, сегментации и извлечения цветов.
- `app.py`: Веб-интерфейс Gradio.
- `API.py`: FastAPI-эндпоинт.
- `data/colors.csv`: База цветов (формат: `colorname,name,hex`).
- `models/`: Папка для моделей YOLO и SAM (см. `models/README.md`).
- `notebooks/step_by_step_processing.ipynb`: Jupyter Notebook с разбором пайплайна.
- `requirements.txt`: Зависимости Python.
- `Dockerfile`: Конфигурация Docker для контейнерного запуска.
