"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.utils import create_logger

import sqlite3 as db
import pandas as pd


#python train_set.py --data_path /extra/dwicke/data/transfer_learning --dataset_type classification --save_dir ../mymodels/finalPRCModels   --epochs 50 --split_type index_predetermined --crossval_index_file /extra/dwicke/data/transfer_learning --excel_file results.xlsx --metric prc-auc


if __name__ == '__main__':
    args = TrainArgs().parse_args()

    names = ['gtpp1', 'gtpp2', 'gtpp3', 'gtpp4', 'gtpp7', 'gtpp8', 'perm']
    
    resultData = {'name': [], 'type': [], 'prc-auc': [], 
        'Runtime': []}
    baseData = args.data_path
    baseSave = args.save_dir
    baseCross = args.crossval_index_file

    for cur_name in names:
        args.data_path = f'{baseData}/{cur_name}/{cur_name}.csv'
        
    
        args.save_dir = f'{baseSave}/{cur_name}_oneinnerfold_notransfer_checkpoints'
        args.crossval_index_file = f'{baseCross}/{cur_name}/outerfold3.pkl' # Going to run on the outer fold 1 and am going to run on the first fold.
        args.no_cuda = False  # these get deleted when modify_train_args gets called so need to reset them
        args.no_features_scaling = False # these get deleted when modify_train_args gets called so need to reset them
        modify_train_args(args)
        args.num_folds = 1 # was told to just do one of the folds.
        logger = create_logger(name=f'train_{cur_name}', save_dir=args.save_dir, quiet=args.quiet)
        mean_score, std_score, mean_time, std_time, numTestSmiles = cross_validate(args, logger)
        resultData['name'].append(cur_name)
        resultData['type'].append(f'No transfer')
        resultData['prc-auc'].append(mean_score)
        resultData['Runtime'].append(mean_time)
            

    # now I'm going to log the results to sqlite so that I can aggregate the results from multiple runs
    # pass the db path as an arg instead
    if args.sqlite_db != 'none':
        engine = db.connect(args.sqlite_db)
        # create the dataframe for storing the results
        df = pd.DataFrame(resultData)
        # then write results using to_sql
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
        df.to_sql('results', con=engine, if_exists='append', index=False)
    elif args.excel_file != 'none':
        df = pd.DataFrame(resultData)
        df.to_excel(f'{baseSave}/{args.excel_file}', index=False)
        
 