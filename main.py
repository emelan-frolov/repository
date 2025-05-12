import os
import time
import logging
import numpy as np
# import cv2 # Не используется напрямую в этой версии main.py
# import rasterio # Используется в analysis.py и io_utils.py
import json # Нужен для сохранения/загрузки результатов анализа и разметки
import subprocess # Нужен для проверки Docker/nvidia-smi
from typing import Optional, Dict, Any, List # Добавили импорты типов
import shutil # Для копирования/перемещения файлов ODM

# Импортируем конфигурацию и модули
import config # Загружаем наш config.py
# Основные рабочие модули для этого пайплайна:
from core import io_utils, analysis, odm_runner
# Вспомогательные функции и логгер:
from utils import helpers

# --- Глобальная настройка логгера ---
# Определяем абсолютный путь к папке вывода для лог-файла
try:
    # Вычисляем абсолютный путь к папке вывода из конфига
    # Папка 'data/output' внутри корня проекта
    abs_output_dir_for_log_and_analysis = os.path.join(config.PROJECT_ROOT, config.OUTPUT_DIR_REL)
    helpers.setup_logging(
        level=config.LOGGING_LEVEL,
        log_to_file=config.LOG_TO_FILE,
        log_filename=config.LOG_FILENAME,
        output_dir=abs_output_dir_for_log_and_analysis # Передаем абсолютный путь
    )
except Exception as log_e:
     # Используем print, так как логгер мог не инициализироваться
     print(f"FATAL: Failed to setup logging - {log_e}", file=sys.stderr) # Используем sys.stderr
     import sys # Импортируем sys здесь, если он нужен
     sys.exit(1) # Завершаемся, если логгер не настроен

# Получаем логгер для этого модуля ПОСЛЕ настройки
logger = logging.getLogger(__name__)

# --- Вспомогательные функции (вынесены из main_pipeline для чистоты) ---

def run_analysis_stage(orthophoto_path: str, output_analysis_dir: str) -> Optional[List[Dict[str, Any]]]:
    """
    Запускает этап анализа парковочных мест.
    """
    if not config.RUN_PARKING_ANALYSIS:
        logger.info("Анализ парковочных мест отключен в конфигурации.")
        return None

    logger.info("--- Этап: Анализ парковочных мест ---")
    analysis_results = []
    with helpers.Timer("Анализ парковочных мест"):
        try:
            vis_path_abs = None
            if config.CREATE_ANALYSIS_VISUALIZATION:
                vis_filename = getattr(config, 'VISUALIZATION_FILENAME', 'analysis_preview.jpg')
                vis_path_abs = os.path.join(output_analysis_dir, vis_filename)

            analysis_results = analysis.analyze_parking_slots(
                orthophoto_path=orthophoto_path,
                confidence_threshold=config.ROBOFLOW_CONFIDENCE_THRESHOLD if config.USE_ROBOFLOW_ANALYSIS else config.LOCAL_CONFIDENCE_THRESHOLD,
                create_visualization=config.CREATE_ANALYSIS_VISUALIZATION,
                visualization_path=vis_path_abs
            )
            if analysis_results is None: analysis_results = []

            if analysis_results:
                results_path_abs = os.path.join(output_analysis_dir, config.ANALYSIS_RESULTS_FILENAME)
                io_utils.save_json(analysis_results, results_path_abs)
        except helpers.AnalysisError as ae:
             logger.error(f"Ошибка во время анализа: {ae}", exc_info=True)
             analysis_results = None
        except Exception as analysis_e:
             logger.error(f"Непредвиденная ошибка во время анализа парковок: {analysis_e}", exc_info=True)
             analysis_results = None
    return analysis_results

def generate_report_stage(stats: dict, output_analysis_dir: str):
    """
    (Опционально) Генерирует текстовый отчет с использованием LLM.
    """
    # ... (Код функции generate_report как в предыдущей версии,
    #      используя output_analysis_dir для сохранения отчета) ...
    # Заглушка, чтобы не повторять большой блок кода
    if not config.USE_LLM_ASSISTANT:
        logger.info("Генерация отчета LLM отключена.")
        return
    logger.info("Генерация отчета LLM (заглушка)...")
    # Реальный код генерации отчета здесь


# --- Основной Пайплайн ---
def main_pipeline():
    """ Основной пайплайн обработки. """
    global_start_time = time.time()
    logger.info("=" * 60)
    logger.info("=== ЗАПУСК ПАЙПЛАЙНА СОЗДАНИЯ ОРТОФОТОПЛАНА (ODM) И АНАЛИЗА ===")
    logger.info("=" * 60)

    # --- 1. Определение и подготовка путей ---
    try:
        project_root_abs = config.PROJECT_ROOT
        # Откуда пользователь КЛАДЕТ исходные изображения
        user_input_image_dir_abs = os.path.join(project_root_abs, config.USER_INPUT_IMAGE_DIR_REL)
        # Базовая папка для ВСЕХ выходных данных СКРИПТА (логи, анализ, копия ортофото)
        output_script_dir_abs = os.path.join(project_root_abs, config.OUTPUT_DIR_REL)
        os.makedirs(output_script_dir_abs, exist_ok=True)

        # Папка ПРОЕКТА ODM на хосте (ВНУТРИ output_script_dir_abs)
        # Именно эта папка будет монтироваться в Docker и передаваться ODM как --project-path
        odm_project_path_on_host = os.path.join(output_script_dir_abs, config.ODM_PROJECT_NAME_IN_OUTPUT)
        # Папка для изображений ВНУТРИ папки проекта ODM на хосте
        odm_images_path_on_host = os.path.join(odm_project_path_on_host, config.ODM_IMAGES_SUBDIR_NAME)

        # Создаем папку проекта ODM и папку images внутри нее, если их нет
        # Удаляем старую папку проекта ODM, если она есть, для чистого запуска
        if os.path.exists(odm_project_path_on_host):
            logger.warning(f"Удаление существующей папки проекта ODM: {odm_project_path_on_host}")
            try:
                 shutil.rmtree(odm_project_path_on_host)
            except OSError as e:
                 logger.error(f"Не удалось удалить старую папку ODM: {e}. Возможны проблемы.")
        os.makedirs(odm_images_path_on_host, exist_ok=True) # Создаст и odm_project_path_on_host

    except AttributeError as attr_e:
         logger.fatal(f"Ошибка доступа к настройкам путей в config.py: {attr_e}.")
         return
    except Exception as path_e:
         logger.fatal(f"Ошибка определения или создания путей проекта: {path_e}.")
         return

    # Собираем статистику для отчета
    pipeline_stats = {"start_time": global_start_time}

    # --- 2. Проверка и КОПИРОВАНИЕ входных данных в рабочую папку ODM ---
    source_images = io_utils.list_images(user_input_image_dir_abs)
    pipeline_stats["image_count"] = len(source_images)
    if not source_images:
        logger.fatal(f"Входные изображения не найдены в '{user_input_image_dir_abs}'. Завершение работы.")
        return

    logger.info(f"Копирование {len(source_images)} изображений из '{user_input_image_dir_abs}' в '{odm_images_path_on_host}'...")
    copied_count = 0
    try:
        for img_path in source_images:
            dest_path = os.path.join(odm_images_path_on_host, os.path.basename(img_path))
            shutil.copy2(img_path, dest_path)
            copied_count +=1
        logger.info(f"Скопировано {copied_count} изображений для ODM.")
        if copied_count == 0 and len(source_images) > 0 :
             raise helpers.PipelineError("Не удалось скопировать входные изображения для ODM.")
    except Exception as copy_e:
        logger.fatal(f"Ошибка при копировании изображений в папку ODM: {copy_e}", exc_info=True)
        return

    # --- 3. Запуск ODM ---
    odm_success = False
    try:
        with helpers.Timer("Выполнение OpenDroneMap"):
            odm_success = odm_runner.run_odm(
                odm_project_path_on_host=odm_project_path_on_host, # Передаем папку проекта ODM на хосте
                odm_options=config.ODM_OPTIONS,
                run_method=config.ODM_RUN_METHOD,
                docker_image=config.ODM_DOCKER_IMAGE
            )
    except helpers.OdmError as odm_e:
         logger.fatal(f"Критическая ошибка ODM: {odm_e}")
         return
    except Exception as e:
         logger.fatal(f"Непредвиденная ошибка при запуске ODM: {e}", exc_info=True)
         return

    if not odm_success:
        logger.fatal("ODM завершился неудачно. Дальнейшая обработка невозможна.")
        return

    # --- 4. Поиск и подготовка результатов ODM ---
    logger.info(f"Поиск результатов ODM в папке: {odm_project_path_on_host}")
    orthophoto_path_from_odm, dsm_path_from_odm = io_utils.find_odm_results(odm_project_path_on_host)

    pipeline_stats["ortho_found"] = bool(orthophoto_path_from_odm)
    pipeline_stats["dsm_found"] = bool(dsm_path_from_odm)
    pipeline_stats["odm_resolution"] = config.ODM_OPTIONS.get("orthophoto-resolution", "N/A")

    final_ortho_path_for_analysis_and_report = None

    if orthophoto_path_from_odm:
         # Копируем ортофото из папки результатов ODM в основную папку вывода скрипта (data/output)
         try:
             final_ortho_filename = config.OUTPUT_FILENAME + ".tif" # Имя из конфига
             destination_path = os.path.join(output_script_dir_abs, final_ortho_filename)
             logger.info(f"Копирование ортофотоплана из '{orthophoto_path_from_odm}' в '{destination_path}'...")
             shutil.copyfile(orthophoto_path_from_odm, destination_path)
             logger.info(f"Ортофотоплан скопирован успешно.")
             final_ortho_path_for_analysis_and_report = destination_path
         except Exception as copy_e:
              logger.warning(f"Не удалось скопировать ортофотоплан: {copy_e}. "
                             f"Анализ будет использовать исходный путь ODM: {orthophoto_path_from_odm}")
              if os.path.exists(orthophoto_path_from_odm):
                   final_ortho_path_for_analysis_and_report = orthophoto_path_from_odm
              else:
                   logger.error("Исходный файл ортофото ODM также не найден!")
                   final_ortho_path_for_analysis_and_report = None
    else:
         logger.error("Не удалось найти итоговый ортофотоплан ODM. Анализ невозможен.")

    # --- 5. Анализ парковочных мест ---
    analysis_results_list = None
    if final_ortho_path_for_analysis_and_report:
        analysis_results_list = run_analysis_stage(final_ortho_path_for_analysis_and_report, output_script_dir_abs)
        pipeline_stats["analysis_run"] = True
        pipeline_stats["analysis_results"] = analysis_results_list if analysis_results_list is not None else []
    else:
         logger.info("Пропуск анализа парковочных мест, так как ортофотоплан недоступен.")
         pipeline_stats["analysis_run"] = False
         pipeline_stats["analysis_results"] = None

    # --- Завершение пайплайна ---
    global_end_time = time.time()
    total_time = global_end_time - global_start_time
    pipeline_stats["total_time"] = total_time

    # --- 6. Генерация отчета (Опционально) ---
    generate_report_stage(pipeline_stats, output_script_dir_abs)

    logger.info("=" * 60)
    logger.info(f"=== ПАЙПЛАЙН ЗАВЕРШЕН за {helpers.format_time(total_time)} ===")
    logger.info(f"Исходные результаты ODM находятся в: {odm_project_path_on_host}")
    logger.info(f"Результаты анализа, лог и копия ортофото находятся в: {output_script_dir_abs}")
    logger.info("=" * 60)


# --- Точка входа ---
if __name__ == "__main__":
    logger.info("--- Инициализация Оркестратора ---")
    logger.info(f"Корневая папка проекта: '{config.PROJECT_ROOT}'")
    abs_user_input_dir = os.path.join(config.PROJECT_ROOT, config.USER_INPUT_IMAGE_DIR_REL)
    abs_output_script_dir = os.path.join(config.PROJECT_ROOT, config.OUTPUT_DIR_REL)
    logger.info(f"Каталог пользовательского входа: '{abs_user_input_dir}'")
    logger.info(f"Каталог выхода скрипта (анализ, логи): '{abs_output_script_dir}'")
    logger.info("-" * 40)

    # Проверка Docker перед запуском
    if config.ODM_RUN_METHOD == 'docker':
        logger.info("Проверка доступности Docker...")
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True, text=True)
            logger.info("Docker найден и доступен.")
        except (FileNotFoundError, subprocess.CalledProcessError) as docker_err:
            logger.error(f"Docker не найден или не отвечает: {docker_err}. "
                         f"Убедитесь, что Docker установлен, запущен и доступен из этой среды (WSL).")
            sys.exit(1) # Выход, если Docker критичен и недоступен
        except Exception as docker_e:
             logger.error(f"Непредвиденная ошибка при проверке Docker: {docker_e}")
             # Не выходим, но ODM, вероятно, не запустится

    # Проверка наличия входных изображений
    if not io_utils.list_images(abs_user_input_dir):
        logger.error(f"!!! Входные изображения не найдены в '{abs_user_input_dir}'. "
                     f"Пожалуйста, добавьте изображения в папку '{config.USER_INPUT_IMAGE_DIR_REL}' и перезапустите. !!!")
    else:
        logger.info("Запуск основного пайплайна обработки...")
        try:
            main_pipeline()
        except helpers.PipelineError as pe:
             logger.critical(f"Критическая ошибка пайплайна: {pe}", exc_info=False)
        except Exception as e:
             logger.critical(f"Необработанная фатальная ошибка в main: {e}", exc_info=True)

    logger.info("--- Завершение работы программы ---")