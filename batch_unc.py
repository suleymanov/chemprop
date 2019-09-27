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
    args = get_training_args()
    pred_args = get_predict_args()
    
#     names = ['gtpp1_clerhuman', 'gtpp1_clermouse', 'gtpp1_clerrat',
#        'gtpp1_clinthuman', 'gtpp1_clintmouse', 'gtpp1_clintrat',
#        'gtpp2_cyp1a2', 'gtpp2_cyp2c19', 'gtpp2_cyp2c8', 'gtpp2_cyp2c9',
#        'gtpp2_cyp2d6', 'gtpp2_cyp3a4', 'gtpp3_eqsolfassif', 'gtpp3_eqsolph7',
#        'gtpp3_eqsolph2', 'gtpp3_eqsolsgf', 'gtpp4_permmdck',
#        'gtpp4_permmdckefflux', 'gtpp6_ppbhuman', 'gtpp6_ppbmouse',
#        'gtpp6_ppbrat']
    
#     names = ['gtpp1_clerhuman', 'gtpp1_clermouse', 'gtpp1_clerrat',
#        'gtpp1_clinthuman', 'gtpp1_clintmouse', 'gtpp1_clintrat',
#        'gtpp3_eqsolfassif', 'gtpp3_eqsolph7',
#        'gtpp3_eqsolph2', 'gtpp3_eqsolsgf', 'gtpp4_permmdck',
#        'gtpp4_permmdckefflux', 'gtpp6_ppbhuman', 'gtpp6_ppbmouse',
#        'gtpp6_ppbrat']

    
    names = ['gtpp1_clerhuman']
    # only care about the suffix.
    names = [name.split('_')[1] for name in names]
    smogn =  '_smogn' if args.smogn else ''
    
    resultData = {'smiles':[], 'name': [], 'pred': [], 'actual': [], 
        'ale': [], 'epi': [] }
    baseData = args.data_path
    baseSave = args.save_dir
    baseCross = args.crossval_index_file
    
    nofs = args.no_features_scaling
    nocu = args.no_cuda

    for cur_name in names:
        # reset the args!
        args = get_training_args()
        pred_args = get_predict_args()
        
        args.data_path = f'{baseData}/{cur_name}/{cur_name}{smogn}.csv'
        args.save_dir = f'{baseSave}/{cur_name}_oneinnerfold_checkpoints'
        args.crossval_index_file = f'{baseCross}/{cur_name}/outerfold3{smogn}.pkl' # Going to run on the outer fold 1 and am going to run on the first fold.
        args.no_cuda = nocu  # these get deleted when modify_train_args gets called so need to reset them
        args.no_features_scaling = nofs # these get deleted when modify_train_args gets called so need to reset them
        modify_train_args(args)
        args.num_folds = 1 # was told to just do one of the folds.
        logger = create_logger(name=f'train_{cur_name}', save_dir=args.save_dir, quiet=args.quiet)
        cross_validate(args, logger)
        
        # now i'm going to run predict
        # example:
        # --test_path /projects/profiling/ta_projects/dgat2_ppb/dgat2_ppb_std.txt --checkpoint_dir /extra/dwicke/mymodels/ppb_reg_en5_checkpoints --preds_path /extra/dwicke/mymodels/ppb_reg_en5_checkpoints/dgat2_ppb_std_preds.csv --estimate_variance
        
        pred_args.test_path = f'{baseData}/{cur_name}/{cur_name}_predict.csv'
        pred_args.checkpoint_dir = f'{baseSave}/{cur_name}_oneinnerfold_checkpoints'
        pred_args.preds_path = f'{baseSave}/{cur_name}_oneinnerfold_checkpoints/pred_results.csv'
        pred_args.no_cuda = False  # these get deleted when modify_train_args gets called so need to reset them
        modify_predict_args(pred_args)
        make_predictions(pred_args)
        # now going to read in the resulting predict file and append to the resultData
        results = pd.read_csv(pred_args.preds_path)
        results.columns = ['smiles', 'pred', 'ale', 'epi']
        actual = pd.read_csv(pred_args.test_path)
        actual.columns = ['smiles', 'actual']
        # update the results
        resultData['smiles'].extend(results['smiles'])
        resultData['pred'].extend(results['pred'])
        resultData['ale'].extend(results['ale'])
        resultData['epi'].extend(results['epi'])
        resultData['actual'].extend(actual['actual'])
        resultData['name'].extend([cur_name]*actual['actual'].count())

    # now I'm going to log the results to sqlite so that I can aggregate the results from multiple runs
        # going to write it out each time.  That way I can look in...
        if args.excel_file != 'none':
            df = pd.DataFrame(resultData)
            if '.csv' in args.excel_file:
                df.to_csv(f'{args.excel_file}', index=False)
            else:
                df.to_excel(f'{args.excel_file}', index=False)
        
 