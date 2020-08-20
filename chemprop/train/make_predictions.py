import csv
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from .predict import predict
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit


@timeit()
def make_predictions(args: PredictArgs, smiles: List[str] = None) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: SMILES to make predictions on.
    :return: A list of lists of target predictions.
    """
    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    # If features were used during training, they must be used when predicting
    if ((train_args.features_path is not None or train_args.features_generator is not None)
            and args.features_path is None
            and args.features_generator is None):
        raise ValueError('Features were used during training so they must be specified again during prediction '
                         'using the same type of features as before (with either --features_generator or '
                         '--features_path and using --no_features_scaling if applicable).')

    # Update predict args with training arguments to create a merged args object
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args: Union[PredictArgs, TrainArgs]

    print('Loading data')
    if args.smiles and smiles is None:
        smiles = args.smiles
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator
        )
    else:
        full_data = get_data(
            path=args.test_path,
            args=args,
            target_columns=[],
            ignore_columns=[],
            skip_invalid_smiles=False,
            store_row=True
        )

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if full_data[full_index].mol is not None:
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), num_tasks))
    if args.dataset_type == 'regression':
        sum_ale_uncs = np.zeros((len(test_data), num_tasks))
        sum_epi_uncs = np.zeros((len(test_data), num_tasks))
    else:
        sum_ale_uncs, sum_epi_uncs = None, None

    # Partial results for variance robust calculation.
    all_preds = np.zeros((len(test_data), num_tasks, len(args.checkpoint_paths)))
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        # Load model
        model = load_checkpoint(checkpoint_path)
        model_preds, ale_uncs, epi_uncs = predict(
            model=model,
            data_loader=test_data_loader,
            batch_size=args.batch_size,
            scaler=scaler,
            sampling_size=args.sampling_size
        )
        sum_preds += np.array(model_preds)
        if args.dataset_type == 'regression':
            if ale_uncs is not None:
                sum_ale_uncs += np.array(ale_uncs)
            if epi_uncs is not None:
                sum_epi_uncs += np.array(epi_uncs)
            if args.estimate_variance:
                all_preds[:, :, index] = model_preds

    avg_preds = sum_preds / len(args.checkpoint_paths)
    avg_preds = avg_preds.tolist()
    if args.dataset_type == 'regression':
        avg_ale_uncs = sum_ale_uncs / len(args.checkpoint_paths)
        avg_ale_uncs = avg_ale_uncs.tolist()
        avg_epi_uncs = (
            np.var(all_preds, axis=2)
            if args.estimate_variance else
            sum_epi_uncs / len(args.checkpoint_paths)
        )
        avg_epi_uncs = avg_epi_uncs.tolist()
    else:
        avg_ale_uncs, avg_epi_uncs = None, None

    # Save predictions
    assert len(test_data) == len(avg_preds)
    if args.dataset_type == 'regression':
        assert len(test_data) == len(avg_ale_uncs)
        assert len(test_data) == len(avg_epi_uncs)

    print(f'Saving predictions to {args.preds_path}')
    makedirs(args.preds_path, isfile=True)
    # Put Nones for invalid smiles
    full_preds = [None] * len(full_data)
    for i, si in full_to_valid_indices.items():
        full_preds[si] = avg_preds[i]
    avg_preds = full_preds
    test_smiles = full_data.smiles()
    if args.dataset_type == 'regression':
        full_ale_uncs = [None] * len(full_data)
        full_epi_uncs = [None] * len(full_data)
        for i, si in full_to_valid_indices.items():
            full_ale_uncs[si] = avg_ale_uncs[i]
            full_epi_uncs[si] = avg_epi_uncs[i]
        avg_ale_uncs = full_ale_uncs
        avg_epi_uncs = full_epi_uncs

    # Write predictions
    if args.preds_path:
        num_empty = num_tasks * (
            args.multiclass_num_classes
            if args.dataset_type == 'multiclass' else 
            3 if args.dataset_type == 'regression' else 1
        )
        with open(args.preds_path, 'w') as f:
            writer = csv.writer(f)
            header = ['smiles']
            if args.dataset_type == 'multiclass':
                for name in task_names:
                    for i in range(args.multiclass_num_classes):
                        header.append(name + '_class' + str(i))
            else:
                header.extend(task_names)
                if args.dataset_type == 'regression':
                    header.extend([tn + '_ale_unc' for tn in task_names])
                    header.extend([tn + '_epi_unc' for tn in task_names])
            writer.writerow(header)
            for i in range(len(avg_preds)):
                row = [test_smiles[i]]
                if avg_preds[i] is not None:
                    if args.dataset_type == 'multiclass':
                        for task_probs in avg_preds[i]:
                            row.extend(task_probs)
                    else:
                        row.extend(avg_preds[i])
                        if args.dataset_type == 'regression':
                            row.extend(avg_ale_uncs[i])
                            row.extend(avg_epi_uncs[i])
                else:
                    row.extend([''] * num_empty)
                writer.writerow(row)
    return avg_preds, avg_ale_uncs, avg_epi_uncs


def chemprop_predict() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    make_predictions(args=PredictArgs().parse_args())
