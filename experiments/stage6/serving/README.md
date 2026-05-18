# Serving-эксперименты

Serving-вариант запускает разработанную локальную model serving-систему с
изолированным рабочим каталогом для каждого экспериментального прогона.

Для каждого запуска раннер задает отдельные переменные окружения:

```text
SERVING_MODEL_ROOT=<каталог прогона>/proposed/models
SERVING_CONFIG_PATH=<каталог прогона>/proposed/config/active_models.json
SERVING_OBSERVABILITY_RING_PATH=<каталог прогона>/proposed/logs/observability.ring
```

Основные сценарии:

```bash
bash scripts/run_single_experiment.sh latency
bash scripts/run_single_experiment.sh resources
bash scripts/run_single_experiment.sh update
bash scripts/run_single_experiment.sh rollback
bash scripts/run_single_experiment.sh recovery
bash scripts/run_single_experiment.sh diagnostics
```

В отчетах результаты serving-варианта лежат в секции `result.proposed`.
Сценарий `diagnostics` фиксирует endpoints наблюдаемости и составляющие времени
запроса, которые используются при анализе работы системы.
