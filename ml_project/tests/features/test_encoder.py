import pandas as pd
import numpy as np

from src.features.encoder import MeanEncoder


def test_mean_encoder():
    df = pd.DataFrame(
        [[1, 1, 12], [1, 0, 101], [0, 1, 25], [1, 2, 16]],
        columns=['target', 'cat_col', 'info']
    )
    correct = [[0.5, 1.0], [1.0, 1.0], [0.5, 0.0], [1.0, 1.0]]
    encoder = MeanEncoder(alpha=0)
    encoder.fit(df[['cat_col', 'info']], df['target'])
    assert isinstance(encoder.transform(df[['cat_col', 'info']]), pd.DataFrame)
    for (_, row), correct_line in zip(encoder.transform(df[['cat_col', 'info']]).iterrows(), correct):
        assert np.allclose(row['cat_col'], correct_line[0])
        assert np.allclose(row['info'], correct_line[1])
