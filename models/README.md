# Model checkpoints

Веса моделей не входят в репозиторий и игнорируются Git.

## YOLOv8 Nano

Проект сохраняет исходную конфигурацию исследования — `yolov8n.pt`.
Ultralytics автоматически загрузит checkpoint при первом создании модели, если
передать имя файла, либо его можно скачать явно:

```bash
curl -L \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt \
  -o models/yolov8n.pt
```

## Segment Anything ViT-B

Runtime настроен на более компактный вариант SAM ViT-B:

```bash
curl -L \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o models/sam_vit_b_01ec64.pth
```

Итоговая структура:

```text
models/
├── README.md
├── yolov8n.pt
└── sam_vit_b_01ec64.pth
```

Для другого SAM-backbone одновременно измените
`CAR_COLOR_SAM_CHECKPOINT` и `CAR_COLOR_SAM_MODEL_TYPE`.

## Лицензии

- Ultralytics распространяет open-source runtime и модели по AGPL-3.0 и
  предлагает отдельную Enterprise License.
- Segment Anything и опубликованные checkpoint-файлы распространяются по
  Apache-2.0.

Перед коммерческим использованием проверьте актуальные условия в исходных
репозиториях разработчиков.
