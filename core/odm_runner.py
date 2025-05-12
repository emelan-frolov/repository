import subprocess
import os
import logging
import shlex
from typing import Dict, Any, Optional
# Импортируем хелперы, чтобы использовать исключение и таймер
from utils import helpers
# Конфиг больше не нужен здесь напрямую, все пути и параметры передаются

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)

def run_odm(
            odm_project_path_on_host: str, # Абсолютный путь к папке проекта ODM на хосте (напр., data/output/odm_processing)
                                           # Ожидается, что она УЖЕ СОДЕРЖИТ подпапку 'images' с входными файлами.
            odm_options: Optional[Dict[str, Any]] = None,
            run_method: str = 'docker',
            docker_image: str = 'opendronemap/odm:latest'
            ):
    """
    Запускает OpenDroneMap. Ожидает, что папка odm_project_path_on_host
    уже содержит подпапку 'images' с входными изображениями.
    Результаты будут записаны также в odm_project_path_on_host.
    Имя проекта для ODM будет извлечено из имени папки odm_project_path_on_host.

    Args:
        odm_project_path_on_host: Абсолютный путь к папке на хосте, которая будет
                                      корнем проекта для ODM. Она должна содержать подпапку 'images'.
                                      ODM будет писать свои результаты в эту же папку.
        odm_options: Словарь с параметрами для ODM (ключ без '--').
        run_method: 'docker' или 'native'.
        docker_image: Имя Docker образа ODM.

    Returns:
        True при успехе (код возврата 0 и папка odm_orthophoto создана), False иначе.

    Raises:
        helpers.PipelineError, helpers.OdmError, ValueError
    """
    project_name_for_log_and_odm = os.path.basename(odm_project_path_on_host) # Имя из пути
    logger.info(f"--- Запуск OpenDroneMap для проекта '{project_name_for_log_and_odm}' (метод: {run_method}) ---")
    logger.info(f"    Папка проекта ODM на хосте (с подпапкой 'images'): {odm_project_path_on_host}")

    # --- Проверки ---
    images_subdir_on_host = os.path.join(odm_project_path_on_host, 'images') # Стандартное имя для ODM
    if not os.path.isdir(images_subdir_on_host) or not os.listdir(images_subdir_on_host):
         logger.error(f"Папка 'images' не найдена или пуста внутри '{odm_project_path_on_host}'.")
         raise helpers.PipelineError(f"'images' subdir not found or empty in ODM project path: {odm_project_path_on_host}")
    if not os.path.isdir(odm_project_path_on_host): # Проверка, что сама папка проекта ODM существует
         logger.error(f"Папка проекта ODM '{odm_project_path_on_host}' не существует.")
         raise helpers.PipelineError(f"ODM project path does not exist: {odm_project_path_on_host}")

    # --- Подготовка команды ---
    cmd = []
    if run_method == 'docker':
        logger.debug(f"Использование Docker образа: {docker_image}")
        cmd = ['docker', 'run', '--rm'] # -it флаги не нужны для неинтерактивного запуска
        try:
             subprocess.run(['docker', '--version'], check=True, capture_output=True, text=True)
             logger.info("Docker доступен.")
        except (FileNotFoundError, subprocess.CalledProcessError) as docker_e:
             logger.fatal("Docker не найден или не запущен. Установите Docker и запустите его.")
             raise helpers.OdmError("Docker is not available.") from docker_e

        # --- Монтирование тома ---
        # Монтируем всю папку проекта ODM (которая содержит 'images' и куда будут писаться результаты)
        # в /code/project_dir внутри контейнера.
        host_odm_project_docker = odm_project_path_on_host.replace('\\', '/')
        cmd.extend(['-v', f'{host_odm_project_docker}:/code/project_dir'])
        logger.info(f"Монтирование тома: Хост='{odm_project_path_on_host}' -> Контейнер='/code/project_dir'")
        # -------------------------

        # --- Обработка GPU ---
        use_gpu_flag = odm_options and odm_options.get('use-gpu', False)
        if use_gpu_flag:
            logger.info("Запрос использования GPU для ODM в Docker.")
            try:
                 subprocess.run(['nvidia-smi'], check=True, capture_output=True, text=True)
                 cmd.extend(['--gpus', 'all'])
                 logger.info("Добавлен флаг --gpus all.")
            except (FileNotFoundError, subprocess.CalledProcessError):
                 logger.warning("Команда 'nvidia-smi' не найдена или вернула ошибку в WSL. GPU не будет использоваться Docker.")
        # ----------------------------------------------------

        # Имя Docker образа
        cmd.append(docker_image)

        # --- АРГУМЕНТЫ ДЛЯ ODM run.py ---
        # 1. Указываем --project-path ВНУТРИ контейнера.
        cmd.extend(['--project-path', '/code/project_dir']) # ODM будет работать внутри этой папки
        logger.info("ODM --project-path (внутри контейнера): /code/project_dir")
        # ODM автоматически найдет /code/project_dir/images

        # 2. Добавляем остальные опции ODM из словаря
        if odm_options:
            options_to_add = odm_options.copy()
            # Удаляем опции, которые управляются иначе или могут конфликтовать
            options_to_add.pop('use-gpu', None)
            # Удаляем параметры, которые больше не нужны или управляются структурой
            options_to_add.pop('name', None)
            options_to_add.pop('project-name', None)
            options_to_add.pop('orthophoto-tif', None)

            for key, value in options_to_add.items():
                arg_key = f'--{key}'
                if isinstance(value, bool):
                    if value: cmd.append(arg_key)
                elif value is not None: cmd.extend([arg_key, str(value)])

        # 3. ПОЗИЦИОННЫЙ АРГУМЕНТ ИМЕНИ ПРОЕКТА БОЛЬШЕ НЕ НУЖЕН,
        #    так как --project-path теперь указывает на конкретную папку проекта,
        #    и ODM будет использовать имя этой папки для своих нужд (если требуется).
        #    По умолчанию ODM создает структуру ВНУТРИ project-path.
        # cmd.append(project_name_for_log_and_odm) # УБРАЛИ позиционный аргумент
        logger.info(f"ODM будет использовать '{project_name_for_log_and_odm}' как имя проекта из пути.")
        # ---------------------------------------------------------

    elif run_method == 'native':
        # --- Логика для нативного запуска (ТРЕБУЕТ АДАПТАЦИИ!) ---
        # Предполагаем, что odm_project_path_on_host содержит images и является project_path
        odm_run_script = '/path/to/your/OpenDroneMap/run.py' # ЗАМЕНИТЬ!
        logger.warning(f"Используется нативный запуск ODM. Убедитесь, что ODM установлен и путь '{odm_run_script}' корректен.")
        if not os.path.exists(odm_run_script):
             logger.fatal(f"Скрипт ODM 'run.py' не найден: {odm_run_script}")
             raise helpers.OdmError(f"ODM run script not found: {odm_run_script}")

        cmd = ['python', odm_run_script]
        cmd.extend(['--project-path', os.path.abspath(odm_project_path_on_host)])

        if odm_options:
             options_to_add = odm_options.copy()
             options_to_add.pop('name', None)
             options_to_add.pop('project-name', None)
             options_to_add.pop('orthophoto-tif', None)
             for key, value in options_to_add.items():
                 arg_key = f'--{key}'
                 if isinstance(value, bool):
                     if value: cmd.append(arg_key)
                 elif value is not None: cmd.extend([arg_key, str(value)])
        # Позиционный аргумент имени проекта здесь тоже не нужен, т.к. есть --project-path
        logger.warning("Нативный запуск: убедитесь, что ODM может найти папку 'images' внутри указанного project-path.")
        # -----------------------------------------------------------
    else:
        logger.fatal(f"Неизвестный метод запуска ODM: {run_method}")
        raise ValueError(f"Unsupported ODM run method: {run_method}")

    command_str_log = " ".join(map(shlex.quote, cmd))
    logger.info(f"Итоговая команда запуска ODM:\n{command_str_log}")

    # --- Запуск ODM и логирование вывода ---
    logger.info("Запуск процесса ODM...")
    return_code = -1
    try:
        working_directory = None # Для Docker cwd не нужен, он работает внутри контейнера
        if run_method == 'native':
            # Для native запуска, возможно, нужно быть в папке проекта ODM
            # или убедиться, что скрипт run.py сам правильно определяет пути.
            # Пока оставляем None, но это может потребовать настройки.
            # working_directory = os.path.dirname(odm_run_script)
            pass

        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace',
                              bufsize=1, universal_newlines=True,
                              cwd=working_directory) as process:
            for line in process.stdout:
                if line:
                    logger.info(f"[ODM] {line.strip()}")
        return_code = process.returncode
    except FileNotFoundError as fnf_e:
        cmd_exec = cmd[0]
        logger.fatal(f"Команда '{cmd_exec}' не найдена. {fnf_e}")
        if cmd_exec == 'docker':
             raise helpers.OdmError("Docker command not found.") from fnf_e
        else:
             raise helpers.OdmError(f"Command '{cmd_exec}' not found.") from fnf_e
    except Exception as e:
        logger.error(f"Ошибка при запуске или во время выполнения ODM: {e}", exc_info=True)
        raise helpers.OdmError(f"Failed to run ODM: {e}") from e

    # --- Проверка результата ---
    if return_code == 0:
        # Проверяем наличие специфичных папок/файлов ODM внутри odm_project_path_on_host
        ortho_folder_check = os.path.join(odm_project_path_on_host, "odm_orthophoto")
        dsm_file_check = os.path.join(odm_project_path_on_host, "odm_dem", "dsm.tif") # Пример

        if os.path.isdir(odm_project_path_on_host) and \
           (os.path.isdir(ortho_folder_check) or os.path.exists(dsm_file_check)): # Хотя бы что-то создалось
            logger.info(f"--- ODM для проекта '{project_name_for_log_and_odm}' завершен успешно ---")
            return True
        else:
            logger.error(f"ODM завершился с кодом 0, но ожидаемые папки/файлы результатов не найдены в: {odm_project_path_on_host}")
            raise helpers.OdmError(f"ODM finished with code 0 but output structure was not found in: {odm_project_path_on_host}")
    else:
         logger.error(f"--- ODM для проекта '{project_name_for_log_and_odm}' завершен с ошибкой (код: {return_code}) ---")
         raise helpers.OdmError(f"ODM process for project '{project_name_for_log_and_odm}' finished with error code {return_code}")

    return False # Не должно достигаться