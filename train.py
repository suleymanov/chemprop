"""Trains a chemprop model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.utils import create_logger, save_metrics


if __name__ == '__main__':
    args = TrainArgs().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    mean_score, std_score = cross_validate(args, logger)
    save_metrics(mean_score, std_score, logger)
