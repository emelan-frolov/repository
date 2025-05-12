import numpy as np
import logging
import os
from typing import List, Dict, Any, Optional
import rasterio # Для чтения ортофото в методе визуализации
import cv2 # Для рисования и чтения/записи превью
# Импортируем SDK Roboflow или нужные ML фреймворки
try:
    from roboflow import Roboflow
    ROBOFLOW_SDK_AVAILABLE = True
except ImportError:
    logging.warning("Пакет roboflow не найден. Анализ через Roboflow API будет недоступен. Установите: pip install roboflow")
    ROBOFLOW_SDK_AVAILABLE = False
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    logging.warning("Пакет ultralytics не найден. Локальный анализ YOLO будет недоступен. Установите: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False
# Импортируем конфиг для параметров
try:
    import config
except ImportError:
    class MockConfig:
        USE_ROBOFLOW_ANALYSIS = True # Пример
        ROBOFLOW_API_KEY = "YOUR_API_KEY"
        ROBOFLOW_PROJECT_ID = "your-project"
        ROBOFLOW_VERSION_ID = "1"
        ROBOFLOW_CONFIDENCE_THRESHOLD = 0.4
        MODELS_DIR_REL = "models"
        LOCAL_MODEL_FILENAME = "yolo.pt"
        LOCAL_CONFIDENCE_THRESHOLD = 0.5
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Примерный корень
    config = MockConfig()
    logging.warning("Не удалось импортировать config.py. Используются значения API/модели по умолчанию/заглушки.")

# Импортируем утилиты
from utils import helpers

logger = logging.getLogger(__name__)

# --- Функция рисования результатов ---
def draw_analysis_results(image_bgr: np.ndarray,
                          predictions: List[Dict[str, Any]],
                          color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
                          line_thickness: int = 2,
                          font_scale: float = 0.5,
                          font_thickness: int = 1):
    """ Рисует bounding box'ы из предсказаний на изображении. """
    vis_image = image_bgr.copy()
    if not predictions:
        return vis_image

    # Цвета по умолчанию (BGR)
    if color_mapping is None:
        color_mapping = {
            # Статусы мест (примеры)
            'occupied': (0, 0, 255), 'not_free': (0, 0, 255),
            'vacant': (0, 255, 0), 'free': (0, 255, 0), 'empty': (0, 255, 0),
            'partially_free': (0, 255, 255),
            # Объекты (примеры)
            'car': (0, 165, 255), 'vehicle': (0, 165, 255),
            'parking_spot': (255, 0, 255), # Фиолетовый для места
            'disabled': (255, 0, 0), # Синий для знака/места инвалида
            'unknown': (128, 128, 128)
        }

    drawn_boxes = 0
    for pred in predictions:
        try:
            # Ожидаем bbox в формате [xmin, ymin, xmax, ymax]
            # или в формате [xc, yc, w, h]
            bbox = pred.get('bbox_pixels') # Предпочитаем готовые пиксельные координаты
            xc, yc, w, h = pred.get('bbox_center_wh', [None]*4)

            if bbox is not None and len(bbox) == 4:
                x_min, y_min, x_max, y_max = map(int, bbox)
            elif None not in [xc, yc, w, h]:
                x_min = int(xc - w / 2)
                y_min = int(yc - h / 2)
                x_max = int(xc + w / 2)
                y_max = int(yc + h / 2)
            else:
                # Пытаемся извлечь из других возможных ключей Roboflow/YOLO
                x, y, width, height = pred.get('x'), pred.get('y'), pred.get('width'), pred.get('height')
                if None not in [x, y, width, height]:
                     x_min = int(x - width / 2); y_min = int(y - height / 2)
                     x_max = int(x + width / 2); y_max = int(y + height / 2)
                else:
                     logger.warning(f"Не удалось извлечь координаты рамки из предсказания: {pred}")
                     continue # Пропускаем это предсказание

            class_name = str(pred.get('status', pred.get('class', 'unknown'))).lower() # Статус или класс
            confidence = pred.get('confidence', 0)
            color = color_mapping.get(class_name, color_mapping['unknown'])

            # Рисуем прямоугольник
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, line_thickness)

            # Добавляем текст (класс/статус и уверенность)
            label = f"{class_name}: {confidence:.2f}"
            (w_text, h_text), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            # Рисуем фон под текстом
            cv2.rectangle(vis_image, (x_min, y_min - h_text - baseline), (x_min + w_text, y_min), color, cv2.FILLED)
            # Рисуем сам текст (белый)
            cv2.putText(vis_image, label, (x_min, y_min - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            drawn_boxes += 1

        except Exception as draw_e:
            logger.error(f"Ошибка при рисовании предсказания {pred}: {draw_e}", exc_info=False)

    logger.info(f"Нарисовано {drawn_boxes} рамок на визуализации.")
    return vis_image

# --- Функции для конкретных методов анализа ---

def _analyze_with_roboflow(orthophoto_path: str, confidence_threshold: float) -> Optional[List[Dict[str, Any]]]:
    """ Выполняет анализ через Roboflow API. """
    if not ROBOFLOW_SDK_AVAILABLE:
        logger.error("Roboflow SDK не установлен.")
        return None

    api_key = getattr(config, 'ROBOFLOW_API_KEY', None)
    project_id = getattr(config, 'ROBOFLOW_PROJECT_ID', None)
    version_id = getattr(config, 'ROBOFLOW_VERSION_ID', None)

    if not api_key or api_key == "ВАШ_СЕКРЕТНЫЙ_API_KEY":
        logger.error("Roboflow API Key не настроен в config.py!")
        return None
    if not project_id or not version_id:
        logger.error("Roboflow project_id или version_id не указаны в config.py.")
        return None

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(project_id)
        model = project.version(int(version_id)).model
        logger.info(f"Подключено к Roboflow: проект '{project_id}', версия {version_id}.")
    except Exception as client_e:
        logger.error(f"Ошибка подключения к Roboflow: {client_e}", exc_info=True)
        return None

    logger.info(f"Отправка изображения '{os.path.basename(orthophoto_path)}' в Roboflow...")
    try:
        results_raw = model.predict(orthophoto_path,
                                     confidence=int(confidence_threshold * 100),
                                     overlap=30) # NMS overlap threshold

        predictions_in = results_raw.json()['predictions']
        logger.info(f"Ответ Roboflow получен. Найдено предсказаний: {len(predictions_in)}")

        # Форматируем результат
        analysis_results = []
        for pred in predictions_in:
             # Копируем основные поля
             res = {
                 "status": pred.get('class', 'unknown'),
                 "confidence": round(pred.get('confidence', 0), 3),
                 "bbox_center_wh": [pred.get('x'), pred.get('y'), pred.get('width'), pred.get('height')]
             }
             # Добавляем пиксельные координаты углов
             x, y, w, h = res["bbox_center_wh"]
             if None not in [x, y, w, h]:
                  res["bbox_pixels"] = [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]
             else:
                  res["bbox_pixels"] = None
             analysis_results.append(res)

        return analysis_results

    except Exception as api_e:
        logger.error(f"Ошибка при выполнении предсказания Roboflow: {api_e}", exc_info=True)
        return None


def _analyze_with_local_yolo(orthophoto_path: str, confidence_threshold: float) -> Optional[List[Dict[str, Any]]]:
    """ Выполняет анализ с помощью локальной модели YOLOv8. """
    if not ULTRALYTICS_AVAILABLE:
        logger.error("Пакет ultralytics не установлен.")
        return None

    # --- Загрузка модели ---
    model_dir_abs = os.path.join(config.PROJECT_ROOT, config.MODELS_DIR_REL)
    model_filename = config.LOCAL_MODEL_FILENAME
    model_path = os.path.join(model_dir_abs, model_filename)

    if not os.path.exists(model_path):
        logger.error(f"Файл локальной модели YOLO не найден: {model_path}")
        return None

    logger.info(f"Загрузка локальной модели YOLO из: {model_path}")
    try:
        model = YOLO(model_path)
        # Пробуем определить устройство (GPU или CPU)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # Импортируем torch, если нужен
        logger.info(f"Локальная модель YOLO загружена. Используется устройство: {device}")
    except NameError: # Если torch не импортирован
         model = YOLO(model_path)
         device = 'cpu'
         logger.info(f"Локальная модель YOLO загружена. Используется устройство: {device} (torch не найден)")
    except Exception as e:
        logger.error(f"Ошибка при загрузке локальной модели YOLO {model_path}: {e}", exc_info=True)
        return None

    # --- Выполнение предсказания ---
    logger.info(f"Запуск предсказания локальной YOLO на: {os.path.basename(orthophoto_path)}...")
    try:
        # Запускаем предсказание. Можно добавить параметры (imgsz, half и т.д.)
        # verbose=False чтобы не дублировать лог YOLO
        results_yolo = model.predict(source=orthophoto_path, conf=confidence_threshold, device=device, verbose=False)

        # Обработка результатов (формат Ultralytics Results)
        analysis_results = []
        # results_yolo - это список (обычно из одного элемента для одного изображения)
        if results_yolo and len(results_yolo) > 0:
            result = results_yolo[0] # Берем результат для первого (и единственного) изображения
            boxes = result.boxes  # Объект Boxes
            class_names = result.names # Словарь {id: name}

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                # Координаты xyxy (xmin, ymin, xmax, ymax)
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                # Координаты xywhn (center_x_norm, center_y_norm, width_norm, height_norm)
                # xywhn = boxes.xywhn[i].cpu().numpy().tolist()
                # Вычисляем центр и размеры в пикселях из xyxy
                x_min, y_min, x_max, y_max = xyxy
                xc = (x_min + x_max) / 2
                yc = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min

                analysis_results.append({
                    "slot_id": class_names.get(cls_id, f"class_{cls_id}"), # Используем имя класса как статус/ID
                    "status": class_names.get(cls_id, f"class_{cls_id}"),
                    "confidence": round(conf, 3),
                    "bbox_center_wh": [xc, yc, w, h],
                    "bbox_pixels": xyxy
                })
        logger.info(f"Предсказание локальной YOLO завершено. Найдено объектов: {len(analysis_results)}")
        return analysis_results

    except Exception as yolo_e:
        logger.error(f"Ошибка при выполнении предсказания локальной YOLO: {yolo_e}", exc_info=True)
        return None


# --- Основная функция анализа (точка входа) ---
def analyze_parking_slots(
    orthophoto_path: str,
    # model и slot_definitions больше не нужны как аргументы, берем из config
    create_visualization: bool = True,
    visualization_path: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Анализирует ортофотоплан для определения статуса парковочных мест,
    используя метод, выбранный в config.py (Roboflow API или локальная YOLO).

    Args:
        orthophoto_path: Путь к GeoTIFF ортофотоплану.
        create_visualization: Создавать ли изображение с визуализацией?
        visualization_path: Путь для сохранения изображения визуализации.

    Returns:
        Список словарей с результатами анализа или None в случае критической ошибки.
    """
    analysis_results = None
    confidence_threshold = 0.4 # Значение по умолчанию

    # Определяем метод анализа из конфига
    use_roboflow = getattr(config, 'USE_ROBOFLOW_ANALYSIS', False)

    if use_roboflow:
        logger.info("Выбран метод анализа: Roboflow API")
        confidence_threshold = getattr(config, 'ROBOFLOW_CONFIDENCE_THRESHOLD', 0.4)
        analysis_results = _analyze_with_roboflow(orthophoto_path, confidence_threshold)
    else:
        logger.info("Выбран метод анализа: Локальная модель YOLO")
        confidence_threshold = getattr(config, 'LOCAL_CONFIDENCE_THRESHOLD', 0.5)
        # Проверяем, нужна ли разметка слотов для локального анализа
        # В текущей реализации _analyze_with_local_yolo она не нужна
        # slot_layout_filename = getattr(config, 'SLOT_LAYOUT_FILENAME', None)
        # if not slot_layout_filename:
        #     logger.warning("Имя файла разметки слотов не указано в config.py для локального анализа.")

        analysis_results = _analyze_with_local_yolo(orthophoto_path, confidence_threshold)
        # Если бы локальной модели требовалась разметка слотов для сопоставления:
        # 1. Загрузить slot_definitions = io_utils.load_json(...)
        # 2. Передать slot_definitions в _analyze_with_local_yolo
        # 3. Внутри _analyze_with_local_yolo сопоставить детекции YOLO со слотами

    # Обработка случая, если анализ не удался
    if analysis_results is None:
         logger.error("Анализ парковочных мест не был выполнен из-за предыдущих ошибок.")
         return [] # Возвращаем пустой список

    # --- Создание и сохранение визуализации ---
    if create_visualization and analysis_results: # Создаем, только если есть результаты
        logger.info("Создание визуализации результатов анализа...")
        vis_create_success = False
        try:
            # Читаем изображение для визуализации
            # Пробуем сначала через rasterio, чтобы учесть геопривязку, если она есть
            img_for_vis = None
            if os.path.exists(orthophoto_path):
                try:
                     with rasterio.open(orthophoto_path) as src:
                          if src.count >= 3:
                               img_rgb = np.transpose(src.read((1,2,3)), (1,2,0))
                               # Нормализуем в uint8
                               max_val = np.iinfo(img_rgb.dtype).max if np.issubdtype(img_rgb.dtype, np.integer) else img_rgb.max()
                               if max_val > 0: img_rgb = (img_rgb / max_val * 255.0).astype(np.uint8)
                               else: img_rgb = img_rgb.astype(np.uint8)
                               img_for_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                          else:
                               logger.warning("Ортофото имеет менее 3 каналов, визуализация будет черно-белой.")
                               img_gray = src.read(1).astype(np.uint8) # Пример чтения первого канала
                               img_for_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                except Exception as rio_vis_e:
                     logger.warning(f"Не удалось прочитать ортофото через Rasterio для визуализации: {rio_vis_e}. Попытка через OpenCV...")

            # Если rasterio не сработал или файл не GeoTIFF, пробуем OpenCV
            if img_for_vis is None:
                img_for_vis = cv2.imread(orthophoto_path, cv2.IMREAD_COLOR)
                if img_for_vis is None:
                     raise ValueError("Не удалось загрузить изображение для визуализации ни через Rasterio, ни через OpenCV.")
                # Убедимся, что 3 канала BGR
                if img_for_vis.ndim == 2:
                     img_for_vis = cv2.cvtColor(img_for_vis, cv2.COLOR_GRAY2BGR)
                elif img_for_vis.ndim == 3 and img_for_vis.shape[2] == 4:
                     img_for_vis = cv2.cvtColor(img_for_vis, cv2.COLOR_BGRA2BGR)


            # Рисуем рамки на основе отфильтрованных результатов
            vis_image = draw_analysis_results(img_for_vis, analysis_results, []) # Передаем analysis_results

            # Определяем путь для сохранения
            vis_path = visualization_path
            if vis_path is None:
                output_analysis_dir = os.path.dirname(orthophoto_path) # Сохраняем рядом с ортофото
                vis_filename = getattr(config, 'VISUALIZATION_FILENAME', 'analysis_preview.jpg')
                vis_path = os.path.join(output_analysis_dir, vis_filename)

            # Сохраняем визуализацию
            success = cv2.imwrite(vis_path, vis_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success:
                logger.info(f"Визуализация сохранена в: {vis_path}")
                vis_create_success = True
            else:
                logger.error(f"Не удалось сохранить визуализацию в {vis_path}")

        except ImportError as ie_vis:
            logger.warning(f"Необходимая библиотека для визуализации не найдена: {ie_vis}. Визуализация не будет создана.")
        except Exception as vis_e:
            logger.error(f"Ошибка создания или сохранения визуализации: {vis_e}", exc_info=True)

    return analysis_results # Возвращаем отфильтрованные результаты