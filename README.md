# ML Project

Перед началом работы необходимо установить зависимости используемые в проекте:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Обучение модели
Запуск обучения: <br>
`python src/run_pipeline.py --config_name {your_config}`

## Датасет
В проекте используется датасет [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)

## Тесты
Запуск тестов: <br>
`python -m pytest --cov src/ tests/`

## Predictions
Получение предсказания для своих данных: <br>
`python -m src.evaluate --model_path {model_path} --transformer_path {transformer_path} --to_predict {data_to_predict}`
