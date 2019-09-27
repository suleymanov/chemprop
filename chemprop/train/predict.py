from typing import List, Tuple, Union

import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            batch_size: int,
            sampling_size: int,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> Tuple[Union[List[List[float]], None], ...]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param batch_size: Batch size.
    :param sampling_size: Sampling size for MC-Dropout.
    :return: A 3-length tuple for predictions, aleatoric uncertainties and epistemic uncertainties.
    Each element is a list of lists. The outer list is examples while the inner list is tasks.
    The second and/or the third element of the tuple can be None if not computed.
    """
    model.eval()

    preds = []
    ale_unc = []
    epi_unc = []

    num_iters, iter_step = len(data_loader), batch_size

    # if aleatoric uncertainty is enabled
    aleatoric = model.aleatoric

    # if MC-Dropout
    mc_dropout = model.mc_dropout

    for batch in tqdm(data_loader, disable=disable_progress_bar):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()

        # Make predictions
        if not aleatoric and not mc_dropout:
            with torch.no_grad():
                batch_preds = model(mol_batch, features_batch)
            batch_preds = batch_preds.data.cpu().numpy()

            # Inverse scaler if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)

        elif aleatoric and not mc_dropout:
            with torch.no_grad():
                batch_preds, batch_logvar = model(mol_batch, features_batch)
                batch_var = torch.exp(batch_logvar)
            batch_preds = batch_preds.data.cpu().numpy()
            batch_ale_unc = batch_var.data.cpu().numpy()

            # Inverse scaler if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                batch_ale_unc = scaler.inverse_transform_variance(batch_ale_unc)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            batch_ale_unc = batch_ale_unc.tolist()
            preds.extend(batch_preds)
            ale_unc.extend(batch_ale_unc)

        elif not aleatoric and mc_dropout:
            with torch.no_grad():
                P_mean = []

                for ss in range(sampling_size):
                    batch_preds = model(mol_batch, features_batch)
                    P_mean.append(batch_preds)

                batch_preds = torch.mean(torch.stack(P_mean), 0)
                batch_epi_unc = torch.var(torch.stack(P_mean), 0)

            batch_preds = batch_preds.data.cpu().numpy()
            batch_epi_unc = batch_epi_unc.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                batch_epi_unc = scaler.inverse_transform_variance(batch_epi_unc)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            batch_epi_unc = batch_epi_unc.tolist()

            preds.extend(batch_preds)
            epi_unc.extend(batch_epi_unc)
            
        elif aleatoric and mc_dropout:
            with torch.no_grad():
                P_mean = []
                P_logvar = []

                for ss in range(sampling_size):
                    batch_preds, batch_logvar = model(mol_batch, features_batch)
                    P_mean.append(batch_preds)
                    P_logvar.append(torch.exp(batch_logvar))

                batch_preds = torch.mean(torch.stack(P_mean), 0)
                batch_ale_unc = torch.mean(torch.stack(P_logvar), 0)
                batch_epi_unc = torch.var(torch.stack(P_mean), 0)

            batch_preds = batch_preds.data.cpu().numpy()
            batch_ale_unc = batch_ale_unc.data.cpu().numpy()
            batch_epi_unc = batch_epi_unc.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                batch_ale_unc = scaler.inverse_transform_variance(batch_ale_unc)
                batch_epi_unc = scaler.inverse_transform_variance(batch_epi_unc)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            batch_ale_unc = batch_ale_unc.tolist()
            batch_epi_unc = batch_epi_unc.tolist()

            preds.extend(batch_preds)
            ale_unc.extend(batch_ale_unc)
            epi_unc.extend(batch_epi_unc)
    if not aleatoric and not mc_dropout:
        return preds, None, None
    elif aleatoric and not mc_dropout:
        return preds, ale_unc, None
    elif not aleatoric and mc_dropout:
        return preds, None, epi_unc
    elif aleatoric and mc_dropout:
        return preds, ale_unc, epi_unc
