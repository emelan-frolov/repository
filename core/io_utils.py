import os
import logging
from typing import List, Tuple,Optional

logger = logging.getLogger(__name__)

IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

def list_images(input_dir: str, supported_formats: Tuple[str] = IMAGE_FORMATS) -> List[str]:
    """ Находит все изображения поддерживаемых форматов в директории. """
    image_files = []
    if not os.path.isdir(input_dir):
        logger.error(f"Директория для поиска изображений не найдена: {input_dir}")
        return []
    try:
        for fname in sorted(os.listdir(input_dir)):
            fpath = os.path.join(input_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(supported_formats):
                image_files.append(fpath)
        logger.info(f"Найдено {len(image_files)} подходящих изображений в '{input_dir}'")
    except OSError as e:
        logger.error(f"Ошибка чтения директории '{input_dir}': {e}")
    return image_files

def find_odm_results(project_output_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Ищет стандартные выходные файлы ODM (ортофото и DSM) в папке проекта.

    Args:
        project_output_path: Путь к папке проекта ODM (например, data/output/odm_processing).

    Returns:
        Кортеж (путь_к_ортофото, путь_к_dsm) или (None, None), если не найдены.
    """
    ortho_path = None
    dsm_path = None

    ortho_folder = os.path.join(project_output_path, "odm_orthophoto")
    dsm_folder = os.path.join(project_output_path, "odm_dem")

    # Ищем ортофото
    if os.path.isdir(ortho_folder):
        ortho_file = os.path.join(ortho_folder, "odm_orthophoto.tif")
        if os.path.exists(ortho_file):
            ortho_path = ortho_file
            logger.info(f"Найден ортофотоплан ODM: {ortho_path}")
        else:
             # Ищем другие возможные имена или расширения, если нужно
             logger.warning(f"Файл odm_orthophoto.tif не найден в {ortho_folder}")
    else:
        logger.warning(f"Папка odm_orthophoto не найдена в {project_output_path}")

    # Ищем DSM
    if os.path.isdir(dsm_folder):
        dsm_file = os.path.join(dsm_folder, "dsm.tif")
        if os.path.exists(dsm_file):
            dsm_path = dsm_file
            logger.info(f"Найден DSM ODM: {dsm_path}")
        else:
            # Ищем другие возможные имена (например, dtm.tif)
            dtm_file = os.path.join(dsm_folder, "dtm.tif")
            if os.path.exists(dtm_file):
                 dsm_path = dtm_file # Используем DTM если DSM нет
                 logger.info(f"Найден DTM ODM (используется как DSM): {dsm_path}")
            else:
                 logger.warning(f"Файлы dsm.tif или dtm.tif не найдены в {dsm_folder}")
    else:
        dsm_in_ortho = os.path.join(ortho_folder, "dsm.tif")
        if os.path.exists(dsm_in_ortho):
             dsm_path = dsm_in_ortho
             logger.info(f"Найден DSM ODM в папке odm_orthophoto: {dsm_path}")
        else:
             logger.warning(f"Папка odm_dem не найдена в {project_output_path}, DSM не найден.")


    return ortho_path, dsm_path

import json

def load_json(file_path: str) -> Optional[dict]:
    """ Загружает данные из JSON файла. """
    if not os.path.exists(file_path):
        logger.error(f"Файл JSON не найден: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON файла '{os.path.basename(file_path)}': {e}")
        return None
    except Exception as e:
        logger.error(f"Ошибка чтения JSON файла '{os.path.basename(file_path)}': {e}", exc_info=True)
        return None

def save_json(data: dict, output_path: str):
    """ Сохраняет данные в JSON файл. """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Данные сохранены в JSON: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения данных в JSON '{output_path}': {e}", exc_info=True)
        return False
