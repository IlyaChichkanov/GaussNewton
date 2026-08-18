# experiments/datasets — сырые данные

Данные **не хранятся в git** (около 215 МБ): каталог целиком под `.gitignore`,
трекается только этот README. На свежем клоне папка будет пустой — ноутбуки из
`../real_data_cars/` упадут с понятным `ValueError: Файл не найден: ...`, пока
файлы не окажутся на месте.

## Что сюда класть

```
datasets/
  CeedEveron.csv                  Kia Ceed, заезд «Эверон» — основной набор
  CeedLateralIntensiveData.csv    Ceed, интенсивная боковая динамика
  KiaCeedLateralIntensiveData.csv то же, ранняя версия
  voyax_free/                     Voyah Free, CAN-логи (~210 МБ)
    GearRatioCheck/{Left,Right,NearZero,all}/
    Lateral Dynamics/{50 kph,80 kph}/
    Skipad/{Left,Right}/
```

В каждом подкаталоге `voyax_free/` лежат CSV, выгруженные из `.blf`
(`EPS_angle.csv`, `ESP_IMU.csv`, `IMU_2.csv`, `ESP_1_rear_speed_wheel.csv`,
`EPS_torgue.csv`), и сам исходный `.blf`.

Каталог `GearRatioCheck/all/` — это склейка `Right` + `Left`, её строит ячейка
в `lateral_movement_voayx_gear_ratio.ipynb`. По умолчанию она пропускается,
если склейка уже есть; пересобрать — `GN_NB_MERGE=1`.

## Где взять

Логи не публичные — забрать из рабочего хранилища экспериментов или у автора
замеров. Rosbag-ноутбуки (`mhe_test_rosbag.ipynb`,
`ceed_new_dynamic_identification.ipynb`) сюда не смотрят: они читают
распарсенные логи из `~/sda_context/logs/ceed/<бэг>/parsed/`.

## Если данные лежат в другом месте

Путь переопределяется переменной окружения, править ноутбуки не нужно:

```bash
GN_DATASETS=/mnt/logs/gaussnewton jupyter lab
```
