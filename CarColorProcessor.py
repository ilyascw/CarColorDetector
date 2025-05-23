from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from sklearn.cluster import KMeans
from matplotlib.colors import rgb_to_hsv
from scipy.stats import mode
import csv

class CarColorProcessor:
    """Класс для обработки изображений автомобилей: детекция, сегментация и определение цветов.

    Attributes:
        yolo_model (YOLO): Модель YOLO для детекции автомобилей.
        sam_model (Any): Модель SAM для сегментации.
        predictor (SamPredictor): Предиктор SAM для генерации масок.
        color_database (str): Путе к базе цветов в формате (csv) [(colorname, name, hex), ...].
        max_size (int): Максимальный размер изображения (для ресайза).
        device (str): Устройство для вычислений ('cpu' или 'cuda').
    """

    def __init__(
        self,
        yolo_model_path: str,
        sam_checkpoint: str,
        sam_model_type: str,
        color_database: str,
        max_size: int = 512,
        device: str = "cpu",
    ) -> None:
        """Инициализация обработчика с моделями и параметрами.

        Args:
            yolo_model_path: Путь к модели YOLO.
            sam_checkpoint: Путь к чекпоинту SAM.
            sam_model_type: Тип модели SAM (например, 'vit_h').
            color_database: Список кортежей [(colorname, name, hex), ...].
            max_size: Максимальный размер изображения (по большей стороне).
            device: Устройство ('cpu' или 'cuda').
        """
        self.max_size = max_size
        self.device = device
        #Загрузка цветов
        try:
            with open(color_database, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self.color_database = [(row['name'], row['description'], row['hex']) for row in reader]
        except FileNotFoundError:
            raise ValueError(f"Color database file not found: {color_database}")
        except Exception as e:
            raise ValueError(f"Error loading color database: {str(e)}")
        # Загрузка YOLO
        try:
            self.yolo_model = YOLO(yolo_model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"YOLO модель не найдена: {yolo_model_path}")

        # Загрузка SAM
        try:
            self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            self.sam_model.to(device)
            self.predictor = SamPredictor(self.sam_model)
        except FileNotFoundError:
            raise FileNotFoundError(f"SAM чекпоинт не найден: {sam_checkpoint}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения: ресайз и нормализация.

        Args:
            image: Входное изображение (BGR, numpy).

        Returns:
            Предобработанное изображение (BGR, numpy).
        """
        h, w = image.shape[:2]
        scale = self.max_size / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    def _detect_cars(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Детекция автомобилей с YOLO.

        Args:
            image: Входное изображение (BGR, numpy).

        Returns:
            Словарь {'car_id': {'bbox': [x_min, y_min, x_max, y_max]}}.
        """
        results = self.yolo_model(image)[0]
        cars = {}
        for i, box in enumerate(results.boxes):
            if self.yolo_model.names[int(box.cls)] == "car":
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                cars[f"car_{i}"] = {"bbox": bbox}
        return cars

    def _segment_cars(self, image: np.ndarray, cars: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Сегментация автомобилей с SAM.

        Args:
            image: Входное изображение (BGR, numpy).
            cars: Словарь с bbox автомобилей.

        Returns:
            Словарь {'car_id': {'bbox': [...], 'mask': np.ndarray}}.
        """
        if not cars:
            return cars

        self.predictor.set_image(image)
        for car_id, car_data in cars.items():
            bbox = car_data["bbox"]
            # Расширение bbox на 10%
            margin = 0.1
            x_min, y_min, x_max, y_max = bbox
            w, h = x_max - x_min, y_max - y_min
            x_min = max(0, x_min - int(w * margin))
            y_min = max(0, y_min - int(h * margin))
            x_max = min(image.shape[1], x_max + int(w * margin))
            y_max = min(image.shape[0], y_max + int(h * margin))
            bbox_expanded = np.array([x_min, y_min, x_max, y_max])

            # Сегментация
            masks, scores, _ = self.predictor.predict(box=bbox_expanded, multimask_output=False)
            if len(masks) == 0 or np.max(scores) < 0.5:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            else:
                mask = masks[np.argmax(scores)]
            cars[car_id]["mask"] = mask

        return cars

    def _detect_colors(self, image: np.ndarray, cars: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Кластеризация цветов автомобилей.

        Args:
            image: Входное изображение (BGR, numpy).
            cars: Словарь с bbox и масками.

        Returns:
            Словарь {'car_id': {'bbox': [...], 'mask': ..., 'colors_rgb': [[r,g,b], ...]}}.
        """
        if not cars:
            return cars

        for car_id, car_data in cars.items():
            mask = car_data["mask"]
            bbox = car_data["bbox"]

            # Извлечение пикселей (BGR → RGB)
            pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[mask > 0].reshape(-1, 3)
            if len(pixels) == 0:
                pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape(-1, 3)

            # Фильтрация в HSV
            hsv_pixels = rgb_to_hsv(pixels / 255.0)
            mask_valid = (hsv_pixels[:, 2] > 0.1) & (hsv_pixels[:, 2] < 0.95)
            filtered_pixels = pixels[mask_valid]

            if len(filtered_pixels) < 10:
                cars[car_id]["colors_rgb"] = []
                continue

            # Кластеризация
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(filtered_pixels)
            labels = kmeans.labels_  # метка кластера для каждого объекта
            # Считаем количество объектов в каждом кластере
            unique_labels, counts = np.unique(labels, return_counts=True)
            # сохраняем в порядке убывания количества пикселей, принадлежащих к этому кластеру
            cars[car_id]['colors_rgb'] = kmeans.cluster_centers_.astype(int)[np.argsort(counts)[::-1]]

        return cars

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Конвертация HEX-цвета в RGB.

        Args:
            hex_color: HEX-строка (например, '#FF0000').

        Returns:
            Кортеж (r, g, b).

        Raises:
            ValueError: Если HEX-формат некорректен.
        """
        try:
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError as e:
            raise ValueError(f"Некорректный HEX-цвет: {hex_color}") from e

    def _get_color_name(self, rgb: np.ndarray) -> str:
        """Определение названия цвета по RGB.

        Args:
            rgb: Массив [r, g, b] (0–255).

        Returns:
            Название цвета (например, 'Белый').
        """
        rgb = np.array(rgb) / 255.0
        hsv = rgb_to_hsv(rgb.reshape(1, 1, 3)).reshape(3)
        hue, saturation, value = hsv

        # Нейтральные цвета
        if saturation < 0.1:
            if value > 0.9:
                return "Белый"
            if value < 0.2:
                return "Черный"
            if value > 0.7:
                return "Светло-серый"
            if value > 0.4:
                return "Серый"
            return "Темно-серый"

        # Поиск ближайшего цвета
        min_dist = float("inf")
        closest_color = None
        for colorname, name, hex_color in self.color_database:
            ref_rgb = np.array(self._hex_to_rgb(hex_color)) / 255.0
            ref_hsv = rgb_to_hsv(ref_rgb.reshape(1, 1, 3)).reshape(3)
            hue_dist = min(abs(hue - ref_hsv[0]), 1 - abs(hue - ref_hsv[0]))
            dist = hue_dist * 0.7 + abs(saturation - ref_hsv[1]) * 0.2 + abs(value - ref_hsv[2]) * 0.1
            if dist < min_dist:
                min_dist = dist
                closest_color = name

        return closest_color

    def _visualize_colors(
        self, image: np.ndarray, cars: Dict[str, Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Визуализация: добавление bbox, полосок цветов и текста.

        Args:
            image: Входное изображение (BGR, numpy).
            cars: Словарь с bbox, масками, цветами.

        Returns:
            Кортеж (выходное изображение, список описаний).
        """
        output_image = image.copy()
        description = []

        if not cars:
            description.append("Автомобили не найдены")
            return output_image, description

        invert_mask = [2, 1, 0]  # BGR → RGB
        for car_id, car_data in cars.items():
            if not car_data.get("colors_rgb", []).size:
                description.append(f"{car_id}: Нет цветов")
                continue

            colors_rgb = car_data["colors_rgb"]
            bbox = car_data["bbox"]

            # Названия цветов
            color_names = [self._get_color_name(color) for color in colors_rgb]
            car_data["color_names"] = color_names

            # Визуализация
            cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
            patch_height = 20
            patch_width = (bbox[2] - bbox[0]) // len(colors_rgb)
            for i, color in enumerate(colors_rgb):
                x_start = bbox[0] + i * patch_width
                x_end = x_start + patch_width
                y_start = bbox[3] + 1
                y_end = y_start + patch_height
                cv2.rectangle(output_image, (x_start, y_start), (x_end, y_end), tuple(color[invert_mask].tolist()), -1)

            # Текст
            text_pos = (bbox[0] + 1, bbox[1] - 1)
            cv2.putText(
                output_image, f"{car_id}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1
            )

            # Описание
            description.append(f"{car_id} цвета:")
            for name, rgb in zip(color_names, colors_rgb):
                description.append(f"Цвет: {name} (RGB: {tuple(rgb.tolist())})")

        return output_image, description

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """Основной метод обработки изображения.

        Args:
            image: Входное изображение (BGR, numpy).

        Returns:
            Словарь {'image': np.ndarray, 'cars': Dict, 'description': List[str]}.

        Raises:
            ValueError: Если изображение некорректно.
        """
        if image is None or image.size == 0:
            raise ValueError("Некорректное изображение")

        # Предобработка
        image = self._preprocess_image(image)

        # Обработка
        cars = self._detect_cars(image)
        cars = self._segment_cars(image, cars)
        cars = self._detect_colors(image, cars)
        output_image, description = self._visualize_colors(image, cars)

        return {
            "image": output_image,
            "cars": cars,
            "description": description,
        }

    def __del__(self) -> None:
        """Освобождение ресурсов."""
        if hasattr(self, "yolo_model"):
            del self.yolo_model
        if hasattr(self, "sam_model"):
            del self.sam_model
        if hasattr(self, "predictor"):
            del self.predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()