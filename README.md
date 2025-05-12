# Orthophoto Analyzer (using OpenDroneMap)

Консольное приложение для автоматического создания ортофотопланов и DSM с помощью OpenDroneMap (ODM), запускаемого в Docker через WSL2, и последующего анализа парковочных мест на полученном ортофотоплане.

## Функционал

*   Автоматизирует запуск полного фотограмметрического пайплайна OpenDroneMap.
*   Использует Docker для запуска ODM, обеспечивая простоту установки и изоляцию зависимостей ODM.
*   Генерирует ортофотоплан (GeoTIFF) и цифровую модель поверхности (DSM GeoTIFF).
*   Запускает пользовательский модуль анализа (`core/analysis.py`) для извлечения информации из ортофотоплана
*   Ведет подробный лог всего процесса.

## Требования к системе

1.  **Docker:**
    *   **Вариант 1: Docker Desktop**. Установите с [официального сайта](https://www.docker.com/products/docker-desktop/) и **убедитесь, что он запущен**.
        *   В Windows Docker Desktop по умолчанию использует WSL 2 бэкенд. Убедитесь, что WSL 2 установлен и включен ([инструкция Microsoft](https://learn.microsoft.com/ru-ru/windows/wsl/install)).
        *   Если ваш проект находится не на диске C, перейдите в настройки Docker Desktop (Settings -> Resources -> File Sharing) и добавьте папку вашего проекта (или весь диск) в список разрешенных для монтирования.
    *   **Вариант 2: Docker Engine в WSL 2** (без Docker Desktop). См. инструкцию по установке ниже.
2.  **Git** (для клонирования репозитория).

## Установка и Настройка

**Выберите один из вариантов настройки среды:**

### Вариант 1: Использование Docker Desktop

1.  **Установите и запустите Docker Desktop.** Убедитесь, что он использует WSL 2 бэкенд (в настройках Docker Desktop -> General).
2.  **Установите Miniconda/Anaconda для Windows.** ([https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html))
3.  **Клонируйте репозиторий:** Откройте **Anaconda Prompt (miniconda3)** или **Anaconda PowerShell Prompt (miniconda3)** из меню Пуск.
    ```powershell
    # Перейдите в папку для проектов
    cd C:\path\to\your\projects
    # Клонируйте
    git clone https://github.com/ilya-orlov-13/orthophoto.git
    cd orthophoto
    ```
4.  **Создайте Conda окружение и установите зависимости:**
    ```powershell
    # Создаем окружение
    conda create -n ortho_analyzer_env python -y
    # Активируем
    conda activate ortho_analyzer_env
    # Устанавливаем зависимости (используя Conda для гео-пакетов)
    conda install -c conda-forge numpy opencv pillow rasterio matplotlib scipy shapely -y
    ```

### Вариант 2: Использование Docker Engine в WSL 2 (Без Docker Desktop)

1.  **Установите WSL 2 и дистрибутив Linux (Ubuntu) по [инструкции Microsoft](https://learn.microsoft.com/ru-ru/windows/wsl/install)**
2.  **Установите Docker Engine в WSL:** 
   **Откройте терминал WSL (Ubuntu)**
   Найдите "Ubuntu" (или имя вашего дистрибутива) в меню "Пуск" и запустите. Все последующие команды выполняются здесь.
   
   ```bash
   # 1. Обновление системы
   sudo apt update && sudo apt upgrade -y
   
   # 2. Установка зависимостей Docker
   sudo apt install -y apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release
   
   # 3. Добавление GPG ключа Docker
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   
   # 4. Добавление репозитория Docker
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     jammy stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   
   # 5. Установка Docker Engine
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   
   # 6. Добавление пользователя в группу docker
   sudo usermod -aG docker $USER
   
   # 7. *** ВАЖНО: Закройте и снова откройте терминал WSL! ***
   
   # 8. Проверка Docker
   docker run hello-world
   # Если видите "Hello from Docker!", все в порядке.
   ```

3.  **Установите Miniconda в WSL:** В **терминале WSL**
   ```bash
   # 1. Скачиваем установщик
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
   
   # 2. Запускаем установку (соглашаемся с лицензией, подтверждаем путь, отвечаем 'yes' на инициализацию)
   bash ~/miniconda.sh -b -p $HOME/miniconda
   
   # 3. Инициализируем Conda для bash (если не сделали на шаге 2)
   ~/miniconda/bin/conda init bash
   
   # 4. Удаляем установщик
   rm ~/miniconda.sh
   
   # 5. *** ВАЖНО: Закройте и снова откройте терминал WSL! ***
   
   # 6. Проверка Conda (в новом терминале должен появиться (base))
   conda info
   ```

4. **Установите [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**
   ```bash
   # В терминале WSL
   # 1.
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   # 2.
   sudo apt-get update
   
   # 3.
   sudo apt-get install -y nvidia-container-toolkit
   
   
   # 4.
   sudo nvidia-ctk runtime configure --runtime=docker
   
   # 5.
   sudo systemctl restart docker
   
   ```

5.  **Клонируйте репозиторий:**
    ```bash
    # В терминале WSL
    # mkdir ~/projects
    # cd ~/projects
    git clone https://github.com/ilya-orlov-13/orthophoto.git
    cd orthophoto
    ```
6.  **Создайте Conda окружение и установите зависимости:**
    ```bash
    # В терминале WSL
    conda create -n ortho_analyzer_env python -y
    conda activate ortho_analyzer_env
    conda install -c conda-forge numpy opencv pillow rasterio matplotlib scipy shapely -y
    ```

## Запуск Приложения

1.  **Убедитесь, что Docker запущен:**
    *   **Docker Desktop:** Запустите Docker Desktop, убедитесь, что он работает (зеленый значок).
    *   **Docker Engine в WSL:** Откройте терминал WSL и проверьте `sudo service docker status`. Если не активен, запустите `sudo service docker start`.
2.  **Откройте терминал:**
    *   **Для Варианта 1 (Docker Desktop):** Откройте **Anaconda Prompt (miniconda3)** или **Anaconda PowerShell Prompt (miniconda3)**.
    *   **Для Варианта 2 (Docker в WSL):** Откройте **терминал WSL (Ubuntu)**.
3.  **Перейдите в корневую папку проекта:**
    *   **Windows Prompt:** `cd C:\path\to\OrthophotoAnalyzer`
    *   **WSL:** `cd /mnt/c/path/to/OrthophotoAnalyzer`
4.  **Активируйте ваше Conda окружение:**
    ```bash
    conda activate ortho_analyzer_env
    ```
5.  **Запустите главный скрипт:**
    ```bash
    python main.py
    ```

## Ожидаемый Результат

*   Скрипт запустит ODM в Docker контейнере. Следите за логами в терминале. Процесс может занять много времени.
*   Результаты ODM будут сохранены в папке `<корень_проекта>/data/output/odm_processing`.
*   Ортофотоплан будет скопирован в `data/output/`.
*   Логи всего процесса будут в `data/output/orthophoto_analyzer.log`.

## Структура Проекта

```
./
├── images/                                      # <-- Входные снимки
├── data/
│   ├── output/odm_processing/images             # <-- Входные снимки
│   └── parking_layout/                          
├── models/                                      
├── core/                                        # Модули ядра (ODM runner, анализ, IO)
├── utils/                                       # Вспомогательные утилиты (логирование)
├── main.py                                      # Главный скрипт
├── config.py                                    # Конфигурация
├── requirements.txt                             # Python зависимости (для pip)
└── README.md               
```

## Устранение Неисправностей

*   **Ошибки Docker:** Убедитесь, что Docker (Desktop или Engine в WSL) запущен. Проверьте настройки File Sharing (для Docker Desktop), если данные не на диске C:. Проверьте правильность монтирования томов в логах `main.py`. Если используете GPU, убедитесь в правильной настройке (NVIDIA Container Toolkit для WSL или настройки в Docker Desktop).
*   **Ошибки ODM:** Смотрите логи с префиксом `[ODM]` в консоли/файле. Часто проблемы связаны с качеством изображений, перекрытием или параметрами ODM в `config.py`.
*   **Ошибки Python/Conda:** Проверьте активацию правильного окружения и установку всех зависимостей (`conda list`).
