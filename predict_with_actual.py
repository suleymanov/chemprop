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




if __name__ == '__main__':
    pred_args = parse_predict_args()
    make_predictions(pred_args)
    # now going to read in the resulting predict file and append to the resultData
    results = pd.read_csv(pred_args.preds_path)
    results.columns = ['smiles', 'pred', 'ale', 'epi']
    
    if pred_args.append_actual:
        actual = pd.read_csv(pred_args.test_path)
        actual.columns = ['smiles', 'actual']
    
        resultData = {'smiles':[], 'pred': [], 'actual': [], 
            'ale': [], 'epi': [] }
        # update the results
        resultData['smiles'].extend(results['smiles'])
        resultData['pred'].extend(results['pred'])
        resultData['ale'].extend(results['ale'])
        resultData['epi'].extend(results['epi'])
        resultData['actual'].extend(actual['actual'])
    
        pd.DataFrame.from_dict(resultData).to_csv(f'{pred_args.excel_file}', index=False)


 