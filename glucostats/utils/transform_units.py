import pandas as pd
from glucostats.utils.format_verification import glucose_data_verification


def mmol_mgdl(df_signals: pd.DataFrame, initial: str, final: str) -> pd.DataFrame:
    """
    Transform glucose from mmol/L to mg/dL and the contrary.

    Parameters
    ----------
    df_signals : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    initial : str, 'mmol' or 'mgdl'
        Initial units of glucose signals from df_signals.

    final : str, 'mmol' or 'mgdl'
        Desired units for glucose signals.

    Returns
    -------
    df_signals : pd.DataFrame MultiIndex
        A pd.DataFrame MultiIndex with same format as df_signals and with glucose signals in the final units.
    """
    df_signals = glucose_data_verification(df_signals)
    column_name_timestamps, column_name_glucose = df_signals.columns
    if initial not in ['mmol', 'mgdl']:
        raise ValueError('Initial value must be either "mmol" or "mgdl"')
    if final not in ['mmol', 'mgdl']:
        raise ValueError('Final value must be either "mmol" or "mgdl"')

    if initial == 'mmol' and final == 'mgdl':
        df_signals[column_name_glucose] = df_signals[column_name_glucose] * 18
    elif initial == 'mgdl' and final == 'mmol':
        df_signals[column_name_glucose] = df_signals[column_name_glucose] / 18
    else:
        df_signals = df_signals

    return df_signals
