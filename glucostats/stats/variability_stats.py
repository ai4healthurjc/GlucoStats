import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from glucostats.utils.format_verification import glucose_data_verification


def glucose_variability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the glucose variability stats:
        - Distance travelled (dt): sum of the absolute difference in glucose levels.
        - Mean Absolute Glucose (mag): the sum of the absolute differences between successive glucose values divided by
          the total time measured in hours.
        - Glycemic Variability Percentage (gvp): calculates the length of the CGM temporal trace by using a
          trigonometric analysis of the data normalized to the duration under evaluation.
        - Coefficient of variation of glucose levels (cv): coefficient of variation of glucose levels in a glucose
          signal.

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    Returns
    -------
    variability_df : pandas.DataFrame
        A dataframe with ids of samples as index and variability stats as columns.
    """
    df = glucose_data_verification(df)
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    glucose_signals = df_copy.groupby(level=0, sort=False)

    std = glucose_signals[column_name_glucose].std()
    mean = glucose_signals[column_name_glucose].mean()

    df_copy['time_diff'] = glucose_signals.apply(
        lambda signal: signal[column_name_timestamps].diff().dt.total_seconds()).values
    df_copy['glucose_diff'] = glucose_signals.apply(
        lambda signal: np.abs(signal[column_name_glucose].diff())).values
    df_copy['distance'] = np.sqrt((df_copy['time_diff']/60)**2 + df_copy['glucose_diff']**2)

    total_time = glucose_signals['time_diff'].sum()
    diff_glucose_total = glucose_signals['glucose_diff'].sum()
    total_distance = glucose_signals['distance'].sum()

    variability_df = pd.DataFrame()
    variability_df['mag'] = diff_glucose_total / (total_time / 3600)
    variability_df['gvp'] = (total_distance / (total_time/60) - 1) * 100
    variability_df['dt'] = diff_glucose_total
    variability_df['cv'] = (std / mean) * 100

    return variability_df


def signal_excursions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the excursions statistics:
        - Mean amplitude of glycemic excursions (mage).
        - Excursion frequency (ef).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    Returns
    -------
    excursions_df : pandas.DataFrame
        A dataframe with ids of samples as index and excursions statistics as columns.
    """
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    glucose_signals_values = df_copy.groupby(level=0, sort=False)[column_name_glucose]

    signals_info = pd.DataFrame()
    signals_info['values'] = glucose_signals_values.apply(lambda x: x.values)
    signals_info['std'] = glucose_signals_values.std()
    signals_info['len'] = glucose_signals_values.apply(len)

    signals_info['peaks'] = glucose_signals_values.apply(lambda x: find_peaks(x)[0])
    signals_info['nadirs'] = glucose_signals_values.apply(lambda x: find_peaks(-x)[0])
    peaks_and_nadirs = signals_info.apply(
        lambda row:
        (list(row['peaks']), list(np.append(row['nadirs'], row['len'] - 1)))
        if len(row['peaks']) > len(row['nadirs']) else
        (list(np.append(row['peaks'], row['len'] - 1)), list(row['nadirs']))
        if len(row['peaks']) < len(row['nadirs']) else
        (list(row['peaks']), list(row['nadirs'])),
        axis=1)
    signals_info['(peaks, nadirs)'] = peaks_and_nadirs

    differences = signals_info.apply(lambda row:
                                     np.abs(row['values'][row['(peaks, nadirs)'][0]] -
                                            row['values'][row['(peaks, nadirs)'][1]]),
                                     axis=1)
    signals_info['excursions'] = differences
    significant_excursions = signals_info.apply(lambda row: row['excursions'][row['excursions'] > row['std']],
                                                axis=1)

    excursions_df = pd.DataFrame()
    excursions_df['mage'] = significant_excursions.apply(lambda x: x.mean() if len(x) > 0 else np.nan)
    excursions_df['ef'] = significant_excursions.apply(len)

    return excursions_df
