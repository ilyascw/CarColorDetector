# Third-party models and datasets

## Runtime models

| Компонент | Назначение | Источник | Условия |
| --- | --- | --- | --- |
| Ultralytics YOLOv8n | Детекция транспорта | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 или Enterprise License |
| Segment Anything ViT-B | Сегментация по bounding box | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | Apache-2.0 |

Checkpoint-файлы не входят в Git-репозиторий и загружаются пользователем
самостоятельно.

## Color catalog

`src/car_color_detector/data/colors.csv` содержит справочник русскоязычных
названий автомобильных оттенков и RGB-приближений. Это эвристический lookup:
результат не следует интерпретировать как заводской код краски.

## Research notebook

`notebooks/step_by_step_processing.ipynb` сохранён как журнал построения
прототипа. Его outputs очищены: репозиторий не использует notebook как источник
метрик или как исполняемый production-контур.
