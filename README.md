[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# Прогнозирование исходов матчей Российской Премьер-лиги (РПЛ)

**Студент:** Елизарова Анастасия Александровна

**Группа:** БИВ233

---

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Данные и источники](#данные-и-источники)
4. [Feature Engineering](#feature-engineering)
5. [Метрика качества](#метрика-качества)
6. [Запуск](#запуск)
7. [Результаты](#результаты)
8. [Отчёт](#отчёт)

---

## Описание задачи

**Задача:** Многоклассовая классификация

**Цель:** Предсказать исход футбольного матча РПЛ с точки зрения хозяев поля:
- `H` — победа хозяев
- `D` — ничья
- `A` — победа гостей

**Датасет:** Матчи РПЛ сезонов **2017–2024** (~2 000 матчей), собранные самостоятельно с нескольких открытых источников.

**Целевая метрика:** **Weighted F1-score**
> Выбрана из-за умеренного дисбаланса классов (~45% H / ~25% D / ~30% A в РПЛ). Weighted F1 учитывает частоту каждого класса и оценивает качество предсказания всех трёх исходов одновременно. Accuracy приводится дополнительно для сравнения с публичными бенчмарками.

---

## Структура репозитория

```
.
├── data
│   ├── raw                         # Сырые данные по источникам и сезонам
│   │   ├── understat_rpl_{year}.csv   # Данные Understat по сезонам
│   │   └── fbref_rpl_{season}.csv     # Данные FBref по сезонам
│   └── processed                   # Очищенные и обработанные данные
│       ├── all_matches.csv         # Полный датасет с фичами
│       ├── train.csv               # Сезоны 2017–2021
│       ├── val.csv                 # Сезон 2022
│       └── test.csv                # Сезоны 2023–2024
├── models                          # Сохранённые модели
├── notebooks
│   ├── 00_data_collection.ipynb   # Парсинг + Feature Engineering + Preprocessing
│   ├── 01_eda.ipynb               # Разведочный анализ данных (EDA)
│   ├── 02_baseline.ipynb          # Baseline-модели (Majority Class, Logistic Regression)
│   └── 03_experiments.ipynb       # Эксперименты: RF, LightGBM, Optuna, ablation study
├── presentation                    # Презентация для защиты
├── report
│   ├── images                     # Графики для отчёта
│   └── report.md                  # Финальный отчёт
├── src
│   ├── __init__.py
│   ├── parsers.py                 # Парсинг Understat + FBref (soccerdata)
│   ├── features.py                # Feature Engineering (xG, форма, Elo, H2H)
│   ├── preprocessing.py           # Предобработка + темпоральный сплит
│   └── models.py                  # Утилиты оценки и визуализации моделей
├── tests
│   └── test.py                    
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Данные и источники

Данные собраны **самостоятельно** из трёх открытых источников:

| Источник | Данные | Метод |
|---|---|---|
| [Understat.com](https://understat.com/league/RPL) | xG, xGA, голы, дата — **все матчи РПЛ 2017–2024** | HTTP-запросы + regex-извлечение JSON из JS |
| [FBref.com](https://fbref.com) | Статистика команд (удары, владение, xG) | `soccerdata` (Python-библиотека) |
| [Sofascore.com](https://www.sofascore.com) | Дополнительная статистика матчей | `soccerdata` |

**Объём датасета:** ~2 000 матчей × ~25 признаков (8 сезонов, 2017–2024)

---

## Feature Engineering

Из сырых данных вычислены следующие группы признаков:

| Группа | Признаки | Окно |
|---|---|---|
| Rolling xG | `home/away_roll_xg_for/against` | 5 матчей |
| Rolling голы | `home/away_roll_goals_for/against` | 5 матчей |
| Форма | `home/away_roll_pts` | 5 матчей |
| Venue-форма | `home/away_venue_form` | только дома/в гостях, 5 матчей |
| Head-to-Head | `h2h_home_win_rate`, `h2h_draw_rate`, `h2h_away_win_rate` | 5 встреч |
| Elo-рейтинг | `home/away_elo`, `elo_diff` | Накопительный |
| Усталость | `home/away_days_rest` | — |
| Позиция в сезоне | `match_week` | — |

**Темпоральный сплит** (по сезонам, без перемешивания):
- **Train:** 2017–2021 — обучение модели
- **Val:** 2022 — подбор гиперпараметров
- **Test:** 2023–2024 — финальная оценка

Случайное перемешивание недопустимо: rolling-фичи и Elo вычисляются по предыдущим матчам, случайный сплит создал бы утечку данных из будущего.

---

## Запуск

### Локально

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Собрать данные и построить фичи
jupyter nbconvert --to notebook --execute notebooks/00_data_collection.ipynb

# 5. EDA
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb

# 6. Baseline-модели
jupyter nbconvert --to notebook --execute notebooks/02_baseline.ipynb

# 7. Эксперименты и финальная модель
jupyter nbconvert --to notebook --execute notebooks/03_experiments.ipynb
```

### Через Docker

```bash
docker-compose up --build
# Jupyter доступен на http://localhost:8888
```

---

## Результаты

| Модель | Weighted F1 (val) | Accuracy (val) | Weighted F1 (test) | Примечание |
|--------|-------------------|----------------|--------------------|------------|
| Majority Class (baseline) | 0.3014 | 0.4708 | 0.2793 | Всегда предсказывает H |
| Logistic Regression | 0.4724 | 0.5375 | 0.4314 | Стандартизация + L2 |
| LightGBM (default) | 0.4692 | 0.4833 | — | 300 итераций |
| Random Forest (default) | 0.5032 | 0.5500 | — | 200 деревьев |
| LightGBM (tuned) | 0.5175 | 0.5667 | — | Optuna, 30 trials |
| **Random Forest (tuned)** | **0.5217** | **0.5750** | **0.4609** | **Лучшая модель, Optuna 30 trials** |

---

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
