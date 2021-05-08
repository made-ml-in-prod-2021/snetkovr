from math import ceil

from src.data.make_dataset import split_train_val_data
from tests.utils import create_fake_dataset


def test_split_train_val_data(validation_config):
    size = 100
    dataset = create_fake_dataset(size=size)
    train, test = split_train_val_data(dataset, validation_config)
    print(train.shape)
    assert train.shape[0] == ceil(size * (1 - validation_config.val_size))
    assert test.shape[0] == ceil(size * validation_config.val_size)
