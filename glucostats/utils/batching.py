import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from glucostats.utils.format_verification import glucose_data_verification

import logging
from colorlog import ColoredFormatter

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)s:%(reset)s %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def extract_batches(df_signals: pd.DataFrame, signals_ids: list or np.array):
    """
    Function to extract the signals of the ids indicated from the signals datafram.

    Parameters
    ----------
    df_signals : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    signals_ids: list or numpy array
        A list of the signals ids to extract.

    Return
    ------
    batch: pd.DataFrame MultiIndex
        Filtered df_signals with just the signals desired.
    """
    return df_signals.loc[signals_ids, :]


def batching(df_signals: pd.DataFrame, batch_size: int = None) -> List[pd.DataFrame]:
    """
    Divide the dataset in batches of size equal to batch_size. In each batch there will be a total of batch_size
    signals. In case the number of signals is not multiple of the batch size, the last batch will contain fewer signals.

    Parameters
    ----------
    df_signals : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    batch_size: int, default None
        If None, no batching is done. If is an integer, it will divide the dataset into batches of size equal
        to batch_size. If the number of signals is not multiple of batch_size, the last batch will contain fewer
        signals.

    Return
    ------
    batches: list of pd.DataFrame MultiIndex
        A list of batches with df_signals format extracted from df_signals of size batch_size. In case the number of
        signals is not multiple of the batch size, the last batch will contain fewer signals.
    """
    df_signals = glucose_data_verification(df_signals)
    if not isinstance(batch_size, int) and batch_size is not None:
        raise TypeError('batch_size must be an integer or None. If None, no batching is done.')
    if isinstance(batch_size, int):
        if batch_size < 1:
            raise ValueError('batch_size must be greater than or equal to 1.')

    batches = []
    if batch_size is not None:
        unique_ids = df_signals.index.unique()
        batches_indexes = np.array_split(unique_ids, np.ceil(len(unique_ids) / batch_size))
        batches = [extract_batches(df_signals, idx_batch)
                   for idx_batch in tqdm(batches_indexes, desc="Batching", unit="batches", ncols=80)]
        logger.info(f'Number of batches: {len(batches)}.')
    else:
        batches.append(df_signals)
        logger.info(f'No batching.')

    return batches
