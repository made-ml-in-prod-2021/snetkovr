import joblib

import click
import pandas as pd

from .features.build_features import make_features


@click.command()
@click.option('--model_path', help='model path', default='models/rf_test.pkl')
@click.option('--transformer_path', help='transformer path', default='models/transformer.pkl')
@click.option('--to_predict', default='test_data.csv')
def evaluate(model_path, transformer_path, to_predict):
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    with open(transformer_path, 'rb') as file:
        transformer = joblib.load(file)

    data = pd.read_csv(to_predict)
    prepared = make_features(transformer, data)
    return model.predict_proba(prepared)


if __name__ == '__main__':
    evaluate()
