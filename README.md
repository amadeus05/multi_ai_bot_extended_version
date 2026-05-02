# v333

Быстрый старт проекта (Windows + venv).

## 1) Создать и активировать виртуальное окружение

```bash
python -m venv .venv
```

PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

CMD:

```bash
.venv\Scripts\activate.bat
```

Git Bash:

```bash
source .venv/Scripts/activate
```

## 2) Установить зависимости

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Настроить `.env`

Скопируй пример:

```bash
copy .env.example .env
```

И заполни нужные поля (минимум):

- `SYMBOLS`
- `TIMEFRAME`
- `START_TS_MS`
- `END_TS_MS`
- `DATA_OUTPUT_DIR`

## 4) Запуск пайплайна датасета

```bash
python runners/run_dataset_pipeline.py
```

Выходные файлы:

- `data/labeled/source=bybit/symbol=.../timeframe=.../dataset.parquet`

## 5) Запуск бэктеста

```bash
python runners/run_backtest_walk_forward.py
```

## 6) Запуск paper/live

Paper:

```bash
python runners/run_paper.py
```

Live:

```bash
python runners/run_live.py
```

## Примечания

- Если PowerShell блокирует активацию скриптов:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

- Если нужен быстрый тест, используй один символ и короткий диапазон времени в `.env`.
