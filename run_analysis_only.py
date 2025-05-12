import os
import sys
import argparse
import logging
import time

# Добавляем корень проекта в путь, чтобы импорты работали
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Импортируем необходимые модули ПОСЛЕ добавления пути
try:
    import config
    from core import analysis, io_utils # io_utils нужен для сохранения JSON
    from utils import helpers
except ImportError as e:
    print(f"FATAL: Ошибка импорта модулей: {e}", file=sys.stderr)
    print("Пожалуйста, убедитесь, что скрипт запускается из корневой папки проекта", file=sys.stderr)
    print("или что структура проекта не нарушена.", file=sys.stderr)
    sys.exit(1)

# --- Настройка логгера ---
# Мы можем использовать ту же настройку, что и в main.py
# Или сделать упрощенную только для консоли
try:
    abs_output_dir_for_log = os.path.join(config.PROJECT_ROOT, config.OUTPUT_DIR_REL)
    helpers.setup_logging(
        level=config.LOGGING_LEVEL,
        log_to_file=config.LOG_TO_FILE,
        log_filename="orthophoto_analyzer_ANALYSIS_ONLY.log", # Другое имя лога
        output_dir=abs_output_dir_for_log
    )
except Exception as log_e:
     print(f"FATAL: Failed to setup logging - {log_e}", file=sys.stderr)
     sys.exit(1)

logger = logging.getLogger(__name__)

# --- Основная функция ---
def run_specific_analysis(orthophoto_file_path: str, output_directory: str):
    """
    Запускает только этап анализа парковок для указанного ортофотоплана.

    Args:
        orthophoto_file_path: Абсолютный или относительный путь к ортофото (.tif).
        output_directory: Папка для сохранения результатов анализа (JSON, виз.).
    """
    start_time = time.time()
    logger.info("=" * 50)
    logger.info("=== ЗАПУСК АНАЛИЗА ПАРКОВОЧНЫХ МЕСТ ===")
    logger.info(f"Входной ортофотоплан: {orthophoto_file_path}")
    logger.info(f"Папка вывода анализа: {output_directory}")
    logger.info("=" * 50)

    # Проверка существования входного файла
    if not os.path.exists(orthophoto_file_path):
        logger.fatal(f"Ошибка: Входной файл ортофотоплана не найден: {orthophoto_file_path}")
        return

    # Создаем папку вывода, если ее нет
    try:
        os.makedirs(output_directory, exist_ok=True)
    except OSError as e:
        logger.fatal(f"Не удалось создать папку вывода '{output_directory}': {e}")
        return

    # Вызываем основную функцию анализа (которая читает конфиг)
    # Передаем абсолютные пути для надежности
    abs_ortho_path = os.path.abspath(orthophoto_file_path)
    abs_output_dir = os.path.abspath(output_directory)

    # Вызываем функцию анализа из main.py или напрямую из analysis.py
    # Вариант 1: Вызов функции из main.py (если она там осталась)
    # (потребуется импортировать main и немного изменить его структуру)

    # Вариант 2: Вызов адаптированной функции из core.analysis
    # Предполагаем, что run_analysis из main.py была вынесена или скопирована
    try:
        # Определяем путь для визуализации
        vis_path_abs = None
        if config.CREATE_ANALYSIS_VISUALIZATION:
            vis_filename = getattr(config, 'VISUALIZATION_FILENAME', 'analysis_preview.jpg')
            vis_path_abs = os.path.join(abs_output_dir, vis_filename)

        # Запуск анализа
        with helpers.Timer("Выполнение анализа парковок"):
             analysis_results = analysis.analyze_parking_slots(
                 orthophoto_path=abs_ortho_path,
                 create_visualization=config.CREATE_ANALYSIS_VISUALIZATION,
                 visualization_path=vis_path_abs
                 # Порог уверенности будет взят из config внутри функции
             )

        # Сохранение JSON результатов (если анализ вернул список)
        if analysis_results is not None:
            results_path_abs = os.path.join(abs_output_dir, config.ANALYSIS_RESULTS_FILENAME)
            save_success = io_utils.save_json(analysis_results, results_path_abs)
            if save_success:
                 logger.info(f"Результаты анализа сохранены в: {results_path_abs}")
            else:
                 logger.error("Не удалось сохранить результаты анализа в JSON.")
        else:
            logger.error("Функция анализа вернула None, возможно, произошла ошибка.")

    except FileNotFoundError as fnf_e:
         logger.fatal(f"Ошибка: Не найдены необходимые файлы (модель, разметка?): {fnf_e}")
    except ImportError as imp_e:
         logger.fatal(f"Ошибка: Отсутствует необходимая библиотека: {imp_e}")
    except Exception as e:
        logger.fatal(f"Непредвиденная ошибка во время анализа: {e}", exc_info=True)

    end_time = time.time()
    logger.info("-" * 50)
    logger.info(f"Анализ завершен за {helpers.format_time(end_time - start_time)}")
    logger.info("-" * 50)


# --- Обработка аргументов командной строки ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск анализа парковочных мест на существующем ортофотоплане.")

    parser.add_argument(
        "orthophoto", # Позиционный аргумент
        type=str,
        help="Путь к входному файлу ортофотоплана (.tif)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None, # По умолчанию None, будем использовать папку ортофото
        help="Папка для сохранения результатов анализа (JSON, визуализация). "
             "По умолчанию - та же папка, где лежит ортофотоплан."
    )
    parser.add_argument(
        "-l", "--loglevel",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default=config.LOGGING_LEVEL, # Берем уровень из конфига по умолчанию
        help="Уровень логирования."
    )

    args = parser.parse_args()

    # Перенастраиваем уровень логирования, если он задан в аргументах
    if args.loglevel != config.LOGGING_LEVEL:
        logger.info(f"Установка уровня логирования на: {args.loglevel}")
        logging.getLogger().setLevel(getattr(logging, args.loglevel))

    # Определяем папку вывода
    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.orthophoto))
        logger.info(f"Папка вывода не указана, используется папка ортофото: {output_dir}")

    # Запускаем анализ
    run_specific_analysis(args.orthophoto, output_dir)

    logger.info("--- Завершение работы скрипта анализа ---")