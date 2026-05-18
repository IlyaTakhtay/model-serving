# Локальная система model serving

## Состав папки

```text
app/                         исходный код сервиса
client/                      клиент для загрузки, активации и отката моделей
config/active_models.json    пример конфигурации активной версии
models/                      рабочий каталог реестра моделей, ONNX-файлы не включены
experiments/detection/       metadata и скрипт загрузки данных для YOLO26n
experiments/                 код воспроизводимых экспериментов
scripts/                     Linux-скрипты запуска, остановки, подготовки и измерений
```

## Установка

Требуется Linux, Python 3.11 или новее, `python3-venv`, а также `curl` или
`wget` для загрузки артефакта.

```bash
cd serving_repo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Зависимости зафиксированы в `requirements.txt` и `pyproject.toml`.

## Загрузка модели

Модельные файлы не хранятся в репозитории. Для экспериментов используется
ONNX-артефакт YOLO26n.

```bash
bash scripts/download_artifacts.sh
```

По умолчанию скрипт скачивает:

```text
https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx
```

Если нужен другой источник:

```bash
MODEL_URL=https://example.org/yolo26n.onnx bash scripts/download_artifacts.sh
```

Ожидаемый путь после загрузки:

```text
experiments/detection/artifacts/yolo26n.onnx
```

## Подготовка активной версии

Чтобы подготовить рабочий каталог модели и файл активной версии:

```bash
bash scripts/prepare_active_model.sh
```

Если артефакт уже лежит в другом месте:

```bash
bash scripts/prepare_active_model.sh /path/to/yolo26n.onnx
```

После выполнения появятся:

```text
models/yolo26n/v1/model.onnx
models/yolo26n/v1/model.json
config/active_models.json
```

`config/active_models.json` содержит активную версию модели:

```json
{
  "yolo26n": {
    "active": "v1",
    "previous": null
  }
}
```

## Запуск сервиса

Запуск:

```bash
bash scripts/start_service.sh
```

Проверка:

```bash
curl http://127.0.0.1:8080/ready
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/runtime/resources
```

Остановка:

```bash
bash scripts/stop_service.sh
```

Логи и временные файлы пишутся в `logs/` и `tmp/`.

## Загрузка модели через API

Активную версию можно подготовить напрямую через
`scripts/prepare_active_model.sh`. Если нужно проверить именно API-загрузку:

```bash
bash scripts/start_service.sh

python client/thin_client.py upload \
  --base-url http://127.0.0.1:8080 \
  --model yolo26n \
  --version v1 \
  --file experiments/detection/artifacts/yolo26n.onnx \
  --metadata experiments/detection/yolo26n.execution.json \
  --activate
```

## Эксперименты

Полный воспроизводимый прогон:

```bash
bash scripts/run_experiments.sh
```

Отдельные сценарии:

```bash
bash scripts/run_single_experiment.sh latency
bash scripts/run_single_experiment.sh resources
bash scripts/run_single_experiment.sh update
bash scripts/run_single_experiment.sh rollback
bash scripts/run_single_experiment.sh recovery
bash scripts/run_single_experiment.sh diagnostics
```

Сценарии покрывают:

- задержку ответа и RPS;
- потребление памяти, CPU и потоков;
- обновление модели;
- откат версии;
- обработку некорректного артефакта и восстановление worker-процесса;
- диагностические endpoints и составляющие времени запроса.

Результаты сохраняются в рабочий каталог экспериментов:

```text
experiments/results/<run-id>/
```

В каждом JSON-отчете сравниваемые варианты разделены:

- `result.baseline` - базовый долгоживущий ONNX Runtime-процесс без слоя model serving;
- `result.proposed` - разработанная model serving-система.

Для сценариев обновления, отката и восстановления отчеты также содержат
шаблоны ручных команд оператора. Они используются для воспроизводимого
сравнения сопровождаемости.

## Baseline и Serving

В папке с экспериментами оставлены два дополнительных README:

```text
baseline/README.md
serving/README.md
```

Они кратко поясняют, что относится к базовому варианту, а что относится к
варианту с разработанной serving-системой.

## Дополнительный датасет

Для основных измерений датасет не нужен: входные тензоры генерируются из
metadata модели. Если нужно скачать датасет для проверки детекции:

```bash
python experiments/detection/download_hf_yolo_dataset.py \
  --repo-id LibreYOLO/road-traffic \
  --output-dir experiments/detection/datasets/road-traffic
```

Скачанные данные игнорируются через `.gitignore`.
