"""Trains a chemprop model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.utils import create_logger, save_metrics


if __name__ == '__main__':
    args = TrainArgs().parse_args()
    mean_score, std_score, _, _, _ = cross_validate(args)
    save_metrics(mean_score, std_score)
