import pandas as pd
from glucostats.utils.format_verification import glucose_data_verification, in_range_verification


def observations_in_ranges(df: pd.DataFrame, in_range_interval: list = [70, 180]) -> pd.DataFrame:
    """
    Calculates the time statistics for specific ranges:
        - Observations in range (n_ir).
        - Observations above range (n_ar).
        - Observations below range (n_br).
        - Observations out range (n_or).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    in_range_interval : list of int|float, default [70, 180]
        Interval defining whether glucose levels are within range or not. The parameter must be a list with the lower
        and upper limit, default is [70, 180] in mg/dL.

    Returns
    -------
    observations_in_ranges_df : pd.DataFrame
        A dataframe with ids of samples as index and observations in ranges as columns.
    """
    df = glucose_data_verification(df)
    in_range_verification(in_range_interval)
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    glucose_signals = df_copy.groupby(level=0, sort=False)

    if 0 not in in_range_interval:
        ranges = [0] + in_range_interval
    else:
        ranges = in_range_interval
    ranges = glucose_signals[column_name_glucose].max().apply(
        lambda x: ranges + [x] if x > ranges[-1] else ranges + [ranges[-1] + 1.])
    df_copy['ranges'] = glucose_signals.apply(
        lambda signal: pd.cut(signal[column_name_glucose], bins=ranges.loc[signal.name])).values

    time_ranges = pd.Series(df_copy['ranges'])
    map_ranges = list(time_ranges.apply(
        lambda interval:
        'n_br' if interval.right <= in_range_interval[0]
        else 'n_ar' if interval.left >= in_range_interval[1]
        else 'n_ir')
    )
    df_copy['ranges'] = map_ranges

    observation_sum = df_copy.groupby([df_copy.index, 'ranges'], sort=False)[column_name_glucose].count()
    observations_in_ranges_df = observation_sum.unstack(level='ranges').fillna(0)

    for stat in ['n_ir', 'n_ar', 'n_br']:
        if stat not in observations_in_ranges_df.columns:
            observations_in_ranges_df[stat] = [0] * len(observations_in_ranges_df)
    observations_in_ranges_df['n_or'] = observations_in_ranges_df['n_ar'] + observations_in_ranges_df['n_br']

    return observations_in_ranges_df


def percentage_observations_in_ranges(df: pd.DataFrame, in_range_interval: list = [70, 180]) -> pd.DataFrame:
    """
    Calculates the time statistics for specific ranges:
        - Percentage observations in range (pn_ir).
        - Percentage observations above range (pn_ar).
        - Percentage observations below range (pn_br).
        - Percentage observations out of range (pn_or).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    in_range_interval : list of int|float, default [70, 180]
        Interval defining whether glucose levels are within range or not. The parameter must be a list with the lower
        and upper limit, default is [70, 180] in mg/dL.

    Returns
    -------
    percentage_observations_in_ranges_df : pd.DataFrame
        A dataframe with ids of samples as index and percentage of observations in ranges as columns.
    """
    observations_in_ranges_df = observations_in_ranges(df, in_range_interval)

    percentage_observations_in_ranges_df = pd.DataFrame()
    total = (observations_in_ranges_df['n_ar'] + observations_in_ranges_df['n_br'] + observations_in_ranges_df['n_ir'])
    percentage_observations_in_ranges_df['pn_ir'] = observations_in_ranges_df['n_ir'] / total * 100
    percentage_observations_in_ranges_df['pn_ar'] = observations_in_ranges_df['n_ar'] / total * 100
    percentage_observations_in_ranges_df['pn_br'] = observations_in_ranges_df['n_br'] / total * 100
    percentage_observations_in_ranges_df['pn_or'] = observations_in_ranges_df['n_or'] / total * 100

    return percentage_observations_in_ranges_df
