import logging
import os
import multiprocessing # Для определения количества ядер CPU

# --- Определение корневой папки проекта ---
# __file__ - это путь к текущему файлу (config.py)
# os.path.dirname() - получает директорию, в которой лежит файл
# os.path.abspath() - получает абсолютный путь
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Основные пути (относительно PROJECT_ROOT) ---
# Папка, КУДА ПОЛЬЗОВАТЕЛЬ КЛАДЕТ ИСХОДНЫЕ СНИМКИ
USER_INPUT_IMAGE_DIR_REL = 'data/input_images'

# Базовая папка для ВСЕХ ВЫХОДНЫХ ДАННЫХ СКРИПТА (логи, анализ, копия ортофото)
# а также внутри нее будет создана папка для ODM
OUTPUT_DIR_REL = 'data/output'

# Имя папки, которую наш main.py создаст ВНУТРИ OUTPUT_DIR_REL
# и передаст ODM как корневую папку его проекта (куда он будет писать результаты
# и где он будет искать подпапку 'images')
ODM_PROJECT_NAME_IN_OUTPUT = "odm_processing"

# Имя стандартной подпапки ВНУТРИ ODM_PROJECT_NAME_IN_OUTPUT, КУДА
# main.py скопирует изображения для ODM и откуда ODM будет их читать.
ODM_IMAGES_SUBDIR_NAME = "images" # Стандартное имя для ODM

# Пути для модуля анализа (остаются относительно PROJECT_ROOT)
PARKING_LAYOUT_DIR_REL = 'data/parking_layout' # Файл с геометрией слотов (если НЕ Roboflow для всего)
MODELS_DIR_REL = 'models'                 # Локальные модели анализа

# --- Параметры запуска ODM ---
ODM_RUN_METHOD = 'docker'                 # Метод запуска: 'docker' или 'native'
ODM_DOCKER_IMAGE = 'opendronemap/odm:latest' # Docker образ ODM

# Параметры, передаваемые в ODM (ключ - параметр ODM без '--')
# В этой схеме НЕ НУЖНО передавать --project-path или имя проекта через опции,
# т.к. odm_runner.py передает --project-path как /code/project_dir
# и ODM использует имя последней папки из этого пути или стандартное имя
ODM_OPTIONS = {
    # --- Основные выходы ---
    "dsm": True,                      # Генерировать DSM?
    "orthophoto-resolution": 5.0,     # Разрешение ортофото в САНТИМЕТРАХ на пиксель!
    # "orthophoto-tif": True,         # Устаревшая/нераспознанная опция, GeoTIFF по умолчанию

    # --- Качество/Скорость ---
    "feature-quality": "medium",
    "pc-quality": "medium",
    "use-gpu": True,                  # Пытаться использовать GPU? (Требует настройки Docker/WSL)
    "max-concurrency": max(1, multiprocessing.cpu_count() // 2),
    "matcher-type": "flann",
    # "resize-to": 2400,              # Опционально: уменьшить изображения ДО обработки
    "fast-orthophoto": False,

    # --- Дополнительные параметры ---
    # "cog": True,                    # Создавать Cloud Optimized GeoTIFF?
    # "build-overviews": True,        # Создавать пирамидные слои?
    "verbose": True,                # Рекомендуется для отладки
    "time": True,                   # Замерять время ODM
    # "use-exif": True,               # Использовать GPS из EXIF для геопривязки? (Часто по умолчанию)
    # "force-gps": True,              # Если хотите принудительно использовать GPS даже при наличии GCP файла
}

# --- Параметры анализа парковок (для analysis.py) ---
RUN_PARKING_ANALYSIS = True
# --- Настройки Roboflow API (если USE_ROBOFLOW_ANALYSIS = True) ---
USE_ROBOFLOW_ANALYSIS = True
# ВАЖНО: НЕ ХРАНИТЕ КЛЮЧ В КОДЕ! Используйте переменные окружения!
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', "ВАШ_СЕКРЕТНЫЙ_API_KEY_ЗДЕСЬ") # ЗАМЕНИТЬ
ROBOFLOW_PROJECT_ID = "car-top-side-dfzt2"     # ID вашего проекта Roboflow
ROBOFLOW_VERSION_ID = "4"              # Номер версии вашей модели Roboflow
ROBOFLOW_CONFIDENCE_THRESHOLD = 0.4

# --- Настройки локальной модели YOLO (если USE_ROBOFLOW_ANALYSIS = False) ---
LOCAL_MODEL_FILENAME = 'yolov8s_parking_best.pt' # Имя файла модели в MODELS_DIR_REL
SLOT_LAYOUT_FILENAME = 'parking_slots_layout.json'# Имя файла разметки в PARKING_LAYOUT_DIR_REL
LOCAL_CONFIDENCE_THRESHOLD = 0.5

# --- Общие настройки анализа ---
# Имя файла для сохранения JSON с результатами анализа (в OUTPUT_DIR_REL)
ANALYSIS_RESULTS_FILENAME = 'parking_analysis_results.json'
# Создавать ли визуализацию результатов анализа?
CREATE_ANALYSIS_VISUALIZATION = True
# Имя файла для визуализации (в OUTPUT_DIR_REL)
VISUALIZATION_FILENAME = 'parking_analysis_preview.jpg'

USE_LLM_ASSISTANT = False
LM_STUDIO_API_BASE = "http://localhost:1234/v1"
LM_STUDIO_MODEL_NAME = "local-model"
LLM_REPORT_PARAMS = {'max_tokens': 350, 'temperature': 0.6}
REPORT_FILENAME = "processing_report.txt" # В папке OUTPUT_DIR_REL

# --- Настройки логирования ---
LOGGING_LEVEL = 'INFO' # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = True
LOG_FILENAME = 'orthophoto_analyzer.log' # В папке OUTPUT_DIR_REL