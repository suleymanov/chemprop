"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions

import pandas as pd

if __name__ == '__main__':
    args = parse_predict_args()

    # going to loop over the fold directories and going to write results
    # jnj	smiles	fold	actual value	predicted value	u_a	u_e
    
    base_test_path = args.test_path
    base_preds_path = args.preds_path
    base_checkpoint_dir = args.checkpoint_dir

    for i in range(20):
        # for each of the folds 0 - 19
        # I first change the args.test_path so that it corresponds with the correct test
        args.test_path = '{}/gtpp6_testfold{}.csv'.format(base_test_path, i // 4 + 1)
        # then I update args.preds_path to point to a new csv file
        args.preds_path = '{}{}.csv'.format(base_preds_path, i)
        # then I need to update the args.checkpoint_dir to point to the correct fold
        args.checkpoint_dir = '{}/fold_{}'.format(base_checkpoint_dir, i)
        make_predictions(args)
        # then load the test path and the preds_path and try and load in the base_pred_path.csv to append and combine to 
        test_df = pd.read_csv(args.test_path)
        preds_df = pd.read_csv(args.preds_path)

        test_df = test_df.drop(columns=['jnj', 'smiles', 'gtpp6_ppbmouse', 'gtpp6_ppbrat'])
        test_df = test_df.rename(columns={'gtpp6_ppbhuman':'actual value'})

        preds_df = preds_df.drop('gtpp6_ppbmouse', 'gtpp6_ppbrat', 'gtpp6_ppbmouse_epi_unc', 'gtpp6_ppbrat_epi_unc', 'gtpp6_ppbmouse_ale_unc', 'gtpp6_ppbrat_ale_unc')

        fold_df = pd.DataFrame({'fold':[i]*len(preds_df.index)})
        new_fold_df = pd.concat([preds_df, test_df, fold_df], axis=1)

        if i == 0:
            # set up the file
            all_pred = pd.DataFrame({'compound_names':[], 'smiles':[], 'gtpp6_ppbhuman': [], 'gtpp6_ppbhuman_ale_unc': [], 'gtpp6_ppbhuman_epi_unc': [], 'actual value': [], 'fold':[]})
            all_pred.to_csv('{}.csv'.format(base_preds_path), index=False)
        
        all_pred = pd.read_csv('{}.csv'.format(base_preds_path))
        # then I'll append the new data and then write it back out
        pd.concat([all_pred, new_fold_df], axis=0).to_csv('{}.csv'.format(base_preds_path), index=False)

         
