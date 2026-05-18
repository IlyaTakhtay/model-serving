# Baseline-эксперименты

Базовый вариант - это долгоживущий локальный процесс на ONNX Runtime без слоя
model serving. Он один раз загружает модель и затем принимает бинарные
тензорные запросы от экспериментального раннера через IPC.

Используемые файлы:

```text
../baseline_daemon.py
../baseline_direct.py
```

Измерения baseline формируются теми же сценариями, что и измерения serving:

```bash
bash scripts/run_single_experiment.sh latency
bash scripts/run_single_experiment.sh resources
bash scripts/run_single_experiment.sh update
bash scripts/run_single_experiment.sh rollback
bash scripts/run_single_experiment.sh recovery
```

В отчетах результаты базового варианта лежат в секции `result.baseline`.
Для обновления, отката и восстановления там же сохраняются шаблоны ручных
операторских команд, использованных для сравнения сопровождаемости.
