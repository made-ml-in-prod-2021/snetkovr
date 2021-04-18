from argparse import ArgumentParser
from os.path import join
import sys
import logging

from sklearn.metrics import roc_auc_score

from data.make_dataset import read_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
N = 5
EDA = 'EDA'


def auto_eda(dataset_path: str, out_file: str, target: str = None) -> None:
    df = read_data(dataset_path)
    logger.info(f"start eda df shape {df.shape}")
    with open(out_file, 'w+') as report_file:
        if target:
            report_file.write(f"target column - {target.__repr__()}\n")
        for column in df.columns:
            report_file.write(f"min value {df[column].min()}\n")
            report_file.write(f"min value {df[column].max()}\n")
            report_file.write(f"unique values {df[column].nunique()}\n")
            report_file.write(f"{df[column].value_counts().nlargest(N)}\n")
            if target and df[column].dtype != str:
                report_file.write(f"raw roc auc {roc_auc_score(df[target], df[column])}\n\n")


def parse_arguments():
    args_parser = ArgumentParser()
    args_parser.add_argument(
        "--path", "-p",
        help="path to dataset",
        type=str
    )
    args_parser.add_argument(
        "--out", "-o",
        help="report file path",
        default="reports",
        type=str
    )
    args_parser.add_argument(
        "--target", "-t",
        help="target value",
        type=str,
        required=False
    )
    return args_parser.parse_args()


def setup_logger():
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


if __name__ == "__main__":
    setup_logger()
    args = parse_arguments()
    auto_eda(args.path, join(args.out, EDA), args.target)
