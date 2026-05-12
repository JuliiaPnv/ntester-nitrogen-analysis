# Система проверки эффективности агротехнологических инструментов

Проект для **проверки эффективности агротехнологических инструментов** (показания прибора, вегетационные индексы) в задачах **прогнозирования урожайности** и **обеспеченности растений макроэлементами** по данным из Excel.

Единая точка входа: **`main_phase.py`** — корреляции, регрессия, классификация (по выбору), сохранение таблиц в Excel и графиков (scatter, predicted vs actual).

## Требования

- Python 3.10+ (рекомендуется 3.12)
- Зависимости из `requirements.txt`

```bash
pip install -r requirements.txt
```

## Формат данных

Входной файл — **Excel (`.xlsx`)**, одна строка = одно наблюдение (точка), столбцы — числовые признаки.

- Фаза в имени столбца задаётся суффиксом **`_1`**, **`_2`**, … (например `N_test_1`, `NDVI_3`).
- **Макроэлемент по фазе** как цель: имя с суффиксом фазы, например `N_1`, `P_2`, `K_1`.
- **Урожайность** как цель: столбец ровно **`yield`** (без суффикса фазы).

Показания прибора перечисляются в `--device-features`, индексы — в `--index-features`. Столбцы вида `N_test_*` указывайте только в `--device-features`, не в `--index-features`.

Перед обучением строки с пропусками или нечисловыми значениями в выбранных столбцах и в `target` удаляются.

## Запуск

Из корня репозитория:

```bash
python main_phase.py --help
```

### Аргументы

| Аргумент | Описание |
|----------|----------|
| `--input` | Путь к `.xlsx` (по умолчанию: `yield_analys.xlsx`) |
| `--target` | Цель: `yield` или макроэлемент с фазой (`N_1`, …) |
| `--device-features` | Один или несколько столбцов прибора |
| `--index-features` | Список индексов (можно пустой список) |
| `--task` | `regression`, `classification` или `both` (по умолчанию: `both`) |
| `--plots-dir` | Каталог для графиков (по умолчанию: `plots/phase`) |
| `--results-dir` | Каталог для Excel-результатов (по умолчанию: `results/phase`) |
| `--random-state` | Seed (по умолчанию: `42`) |

### Примеры

**Макроэлемент (фаза 1):**

```bash
python main_phase.py 
  --input yield_analys.xlsx 
  --target N_1 
  --device-features N_test_1 
  --index-features NDVI_1 GNDVI_1 NDRE_1 CI_1 
  --task both 
  --plots-dir plots/phase 
  --results-dir results/phase
```

**Урожайность (несколько фаз):**

```bash
python main_phase.py 
  --input yield_analys.xlsx 
  --target yield 
  --device-features N_test_1 N_test_2 N_test_3 N_test_4 
  --index-features NDVI_1 GNDVI_1 NDRE_1 CI_1 NDVI_2 GNDVI_2 NDRE_2 CI_2 NDVI_3 GNDVI_3 NDRE_3 CI_3 NDVI_4 GNDVI_4 NDRE_4 CI_4 
  --task both 
  --plots-dir plots/phase 
  --results-dir results/phase
```

## Результаты

При `--results-dir results/phase` и цели `N_1`:

- `results/phase/N_1/correlations_N_1.xlsx` — корреляции Пирсона признаков с целью
- `results/phase/N_1/model_results_N_1.xlsx` — метрики регрессии по наборам признаков и моделям
- `results/phase/N_1_class/model_results_N_1_class.xlsx` — классификация (низкий/высокий уровень по медиане цели), если `--task` включает классификацию

Графики (при регрессии): `plots/phase/scatter_<target>/`, `plots/phase/predictions_<target>/`.

## Структура кода

| Путь | Назначение |
|------|------------|
| `main_phase.py` | CLI и запуск |
| `src/phase_analysis/pipeline.py` | Сценарий `run_analysis`: корреляции, обучение, сохранение |
| `src/phase_analysis/feature_sets.py` | Формирование наборов признаков (макроэлемент / `yield`) |
| `src/phase_analysis/preprocessing.py` | Проверка столбцов, числовое приведение, удаление пропусков |
| `src/phase_analysis/evaluation.py` | Регрессия, CV, метрики |
| `src/phase_analysis/classification.py` | Бинарная классификация по медиане цели |
| `src/phase_analysis/analysis.py` | Корреляции и scatter-графики |
| `src/phase_analysis/models.py` | Набор моделей sklearn |
| `src/phase_analysis/io_utils.py` | Чтение Excel |
