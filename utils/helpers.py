import logging
import sys
import os
import time
from typing import Optional

_logger_initialized = False

def setup_logging(level: str = 'INFO', log_to_file: bool = False, log_filename: str = 'pipeline.log', output_dir: Optional[str] = None):
    """ Настраивает базовое логирование. """
    global _logger_initialized
    if _logger_initialized:
        # Просто меняем уровень, если уже настроено
        try:
            log_level_numeric = getattr(logging, level.upper(), logging.INFO)
            logging.getLogger().setLevel(log_level_numeric)
            logging.info(f"Уровень логирования изменен на: {level}")
        except Exception as e:
            logging.error(f"Не удалось изменить уровень логирования: {e}")
        return

    try:
        log_level_numeric = getattr(logging, level.upper(), logging.INFO)
        log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(log_format, datefmt=date_format)
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level_numeric)
        # Очищаем старые обработчики перед добавлением новых
        if root_logger.hasHandlers():
             root_logger.handlers.clear()

        # Консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Файл
        if log_to_file:
            log_path = log_filename
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    log_path = os.path.join(output_dir, log_filename)
                except OSError as e:
                    root_logger.error(f"Не удалось создать директорию для лог-файла '{output_dir}': {e}. Лог будет сохранен в текущей директории.")
                    log_path = log_filename
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Логирование также настроено в файл: {os.path.abspath(log_path)}")

        root_logger.info(f"Логирование настроено. Уровень: {level}")
        _logger_initialized = True
    except Exception as e:
         print(f"Критическая ошибка при настройке логирования: {e}", file=sys.stderr)

class PipelineError(Exception):
    """ Базовое исключение для ошибок пайплайна. """
    pass

class OdmError(PipelineError):
    """ Ошибка при выполнении ODM. """
    pass

class AnalysisError(PipelineError):
    """ Ошибка на этапе анализа. """
    pass

def format_time(seconds: float) -> str:
    """ Форматирует время. """
    try:
        secs = int(seconds)
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0: return f"{hours:d}:{mins:02d}:{secs:02d}"
        elif mins > 0: return f"{mins:02d}:{secs:02d}"
        else: return f"{secs:02d} сек"
    except Exception: return f"{seconds:.2f} сек"

class Timer:
    """ Контекстный менеджер для замера времени. """
    def __init__(self, message: str = "Время выполнения", log_level=logging.INFO):
        self.message = message
        self.log_level = log_level
        self._start_time = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        logging.log(self.log_level, f"[Timer] Старт: {self.message}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.perf_counter() - self._start_time
        logging.log(self.log_level, f"[Timer] Завершено: {self.message} за {format_time(elapsed_time)} ({elapsed_time:.3f} сек)")