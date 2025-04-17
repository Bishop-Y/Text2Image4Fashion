# Структура
- `config/`:
  - `dataset/` - конфиг для датасета.
  - `gan/` - конфиг для модели.
  - `system/` - конфиг для системы (пример: обучать на GPU или нет).
  - `text/` - конфиг для текста.
  - `training/` - конфиг для параметров обучения.

- `fashion_generator/`:
  - `datamodules/` — модули загрузки данных.
  - `models/` — компоненты модели.
  - `modules/` — обучение.
  - `scripts/` - вспомогательные скрипты.
  - `utils/` - вспомогательная функциональность для обучения.

- `main.py` - точка запуска программы.

- `requirements.txt` - зависимости проекта.
  
## Создание виртуального окружения и установка зависимостей

Рекомендуется использовать виртуальное окружение для управления зависимостями проекта.

### На Linux/MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

### Загрузка DVC‑данных:

В проекте данные управляются с помощью DVC, поэтому после клонирования репозитория необходимо подгрузить данные из удалённого хранилища.

1. Убедитесь, что DVC установлен:
   
   ```bash
   pip install dvc
   ```
  
2. Проверьте конфигурацию DVC:
  
   Откройте файл dvc/config и убедитесь, что он выглядит следующим образом:
   
   ```bash
   [core]
    remote = myremote

   [remote "myremote"]
      url = C:\Users\Tom\DVC
   ```
   При необходимости адаптируйте URL под ваше удалённое хранилище.

3. Скачайте данные:
   
   ```bash
   dvc pull
   ```

### Настройка ClearML:

ClearML используется для отслеживания экспериментов в проекте. Для настройки кредов ClearML выполните следующие действия:

1. Установите ClearML, если он ещё не установлен:

   ```bash
   pip install clearml==1.18.0
   ```

2. Создайте аккаунт на ClearML Server (или используйте существующий) и получите креды (API ключ, секретный ключ, URL сервера).

3. Настройте креды:

   Через переменные окружения:

   На Linux/MacOS:

   ```bash
   export CLEARML_API_HOST="https://your-clearml-server.com"
   export CLEARML_WEB_HOST="https://your-clearml-server.com"
   export CLEARML_API_ACCESS_KEY="YOUR_ACCESS_KEY"
   export CLEARML_API_SECRET_KEY="YOUR_SECRET_KEY"
   ```

   Или через файл конфигурации:

   Создайте файл clearml.conf (например, в домашней директории) и настройте его согласно документации ClearML.

4. Проверьте базовую интеграцию в коде.

   В файле main.py уже должна быть инициализация ClearML:

   ```bash
   from clearml import Task
   task = Task.init(project_name="DeepFashion GAN", task_name="Trying ClearML")
   ```

# Запуск

1. Установить зависимости с помощью команды:
   
   ```bash
   pip install -r requirements.txt
   ```

2. При необходимости настроить конфиги в папке `config`

3. Запустить с помощью команды:
   ```bash
   python main.py
   ```
