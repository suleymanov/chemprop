"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.utils import create_logger

import sqlite3 as db
import pandas as pd


if __name__ == '__main__':
    args = TrainArgs().parse_args()
    if args.quantile_transformer == 'all':
        
        partitions = ['', '_lessThanTen', '_lessThanSeven']
        name = ['(all values)', '(<10)', '(<7)']
        baseData = args.data_path.replace('.csv', '')
        i = 0
        for part in partitions:

            quant_trans = ['none', 'normal', 'uniform']
            baseSave = '{}{}'.format(args.save_dir, part)
            args.data_path = '{}{}.csv'.format(baseData, part)
            cur_name = '{} {}'.format(args.exp_name, name[i])
            i = i + 1 # increment each type
            resultData = {'name': [cur_name], 'RMSE': [], 'RMSE_QuantileTransformer (Gaussian output)': [], 
            'RMSE_QuantileTransformer (uniform output)': [], 'nCompounds_in_test' : []}
            for qt in quant_trans:
                args.save_dir = '{}_{}'.format(baseSave, qt)
                args.quantile_transformer = qt
                logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
                mean_score, std_score, mean_time, std_time, numTestSmiles = cross_validate(args, logger)
                if qt == 'none':
                    resultData['RMSE'].append(mean_score)
                    resultData['nCompounds_in_test'].append(numTestSmiles)
                elif qt == 'normal':
                    resultData['RMSE_QuantileTransformer (Gaussian output)'].append(mean_score)
                elif qt == 'uniform':
                    resultData['RMSE_QuantileTransformer (uniform output)'].append(mean_score)

            # now I'm going to log the results to sqlite so that I can aggregate the results from multiple runs
            # pass the db path as an arg instead
            if args.sqlite_db != 'none':
                engine = db.connect(args.sqlite_db)
                # create the dataframe for storing the results
                df = pd.DataFrame(resultData)
                # then write results using to_sql
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
                df.to_sql('results', con=engine, if_exists='append', index=False)
    else:
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
        mean_score, std_score, mean_time, std_time, numTestSmiles = cross_validate(args, logger)
