import numpy as np
import pandas as pd
from glucostats.utils.format_verification import glucose_data_verification, in_range_verification


def g_control(df: pd.DataFrame, in_range_interval: list = [70, 180], a: int or float = 1.1, b: int or float = 2.0,
              c: int or float = 30, d: int or float = 30) -> pd.DataFrame:
    """
    Calculates the glycemic control indexes:
        - Hypoglycemic Index (hypo_index).
        - Hyperglycemic Index (hyper_index).
        - Index of Glycemic Control (igc).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    in_range_interval : list of int|float, default [70, 180]
        Interval defining whether glucose levels are within range or not. The parameter must be a list with the lower
        and upper limit, default is [70, 180] in mg/dL.

    a: int | float, default 1.1
        Exponent, generally in the range from 1.0 to 2.0. Default value of 1.1.

    b: int | float, default 2.0
        Exponent, generally in the range from 1.0 to 2.0. Default value of 2.0.

    c, d : int | float, default 30
        Scaling factor. Default 30 to display Hyperglycemia Index, Hypoglycemia Index, and IGC on approximately the same
        numerical range as measurements of HBGI, LBGI and GRADE

    Returns
    -------
    control_df : pandas.DataFrame
        A dataframe with ids of samples as index and glycemic control stats as columns.
    """
    df = glucose_data_verification(df)
    in_range_verification(in_range_interval)
    for param in [a, b, c, d]:
        if not (isinstance(param, int) or isinstance(param, float)):
            raise ValueError(f'{param} must be an integer or float')
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    patients_observations = df_copy.groupby(level=0, sort=False)[column_name_glucose].count()

    lltr, ultr = in_range_interval[0], in_range_interval[1]
    low_values = df_copy[column_name_glucose].apply(
        lambda x: (lltr - x) ** b if x < lltr else 0).groupby(level=0, sort=False)
    high_values = df_copy[column_name_glucose].apply(
        lambda x: (x - ultr) ** a if x > ultr else 0).groupby(level=0, sort=False)

    control_df = pd.DataFrame()
    control_df['hypo_index'] = low_values.sum() / (d * patients_observations)
    control_df['hyper_index'] = high_values.sum() / (c * patients_observations)
    control_df['igc'] = control_df['hypo_index'] + control_df['hyper_index']

    return control_df


def a1c_estimation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the a1c estimation by two different methods:
        - Estimated A1C (eA1C).
        - Glucose Management Indicator (gmi).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    Returns
    -------
    a1c_df : pandas.DataFrame
        A dataframe with ids of samples as index and a1c estimations as columns.
    """
    df = glucose_data_verification(df)
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    signals_mean = df_copy.groupby(level=0, sort=False)[column_name_glucose].mean()

    a1c_df = pd.DataFrame()
    a1c_df['gmi'] = 3.31 + 0.02392 * signals_mean
    a1c_df['eA1C'] = (signals_mean + 46.7) / 28.7

    return a1c_df


def qgc_index(df: pd.DataFrame, ideal_bg: int or float = 120) -> pd.DataFrame:
    """
    Calculates the quality of glycemic control indexes:
        - M value (m_value).
        - J index (j_index).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    ideal_bg : int | float, default=120
        The glucose level ideal value. Default 120.

    Returns
    -------
    qgc_df : pandas.DataFrame
        A dataframe with ids of samples as index and quality of glycemic control indexes as columns.
    """
    if not (isinstance(ideal_bg, int) or isinstance(ideal_bg, float)):
        raise ValueError('ideal_bg must be an integer or float')
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    glucose_signals_values = df_copy.groupby(level=0, sort=False)[column_name_glucose]

    m_values = df_copy[column_name_glucose].apply(lambda x: np.abs(10 * np.log10(x / ideal_bg)) ** 3)
    max_difference = glucose_signals_values.max() - glucose_signals_values.min()
    mean_mvalues = m_values.groupby(level=0, sort=False).mean()

    qgc_df = pd.DataFrame()
    qgc_df['m_value'] = mean_mvalues + (max_difference / 20)
    qgc_df['j_index'] = 0.001 * ((glucose_signals_values.mean() + glucose_signals_values.std()) ** 2)

    return qgc_df
