""" Run a batch of uncertainty models. """

from chemprop.parsing import get_training_args
from chemprop.parsing import modify_train_args
from chemprop.parsing import get_predict_args
from chemprop.parsing import modify_predict_args
from chemprop.train import cross_validate
from chemprop.utils import create_logger
from chemprop.train import make_predictions

import sqlite3 as db
import pandas as pd
import torch

from dask.distributed import Client
from dask import delayed

def predict_work(pred_args):
    modify_predict_args(pred_args)
    make_predictions(pred_args)
    


if __name__ == '__main__':

    # the idea is that we pass a folder with the compounds that need predicting
    # --test_path '/projects/profiling/chemlib_janssen/vs_for_chemprop_chemaxon_std/*.txt'
    # --preds_path '/ta_projects/lpar12/results'
    # 
    pred_args = get_predict_args()
    # for now I'm just setting n_workers as fixed
    n_workers = 32
    client = Client(n_workers)

    tasks = []
    i = 0
    for cur_test_path in pathlib.Path(folder).glob(pred_args.test_path):
        pred_args.test_path = cur_test_path
        fname = cur_test_path.split('/')[-1].split('.')[0]
        pred_args.preds_path = f'{pred_args.preds_path}/{fname}_predictions.csv'
        pred_args.gpu = i % torch.cuda.device_count()
        tasks.append(delayed(predict_work)(pred_args))
        i = i + 1
        
    dask.compute(*tasks)
    
    client.close()

        
 