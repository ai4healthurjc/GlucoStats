import pandas as pd
from glucostats.utils.format_verification import glucose_data_verification, in_range_verification


def time_in_ranges(df: pd.DataFrame, in_range_interval: list = [70, 180], time_units: str = 'm') -> pd.DataFrame:
    """
    Calculates the time for specific ranges:
        - Time in range (t_ir).
        - Time above range (t_ar).
        - Time below range (t_br).
        - Time out range (t_or).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    in_range_interval : list of int|float, default [70, 180]
        Interval defining whether glucose levels are within range or not. The parameter must be a list with the lower
        and upper limit, default is [70, 180] in mg/dL.

    time_units : str, default 'm'
        Time units to calculate time in range. Can be 'h' (hours), 'm' (minutes) or 's' (seconds). Default 'm'.

    Returns
    -------
    time_in_ranges_df : pd.DataFrame
        A dataframe with ids of samples as index and time in ranges as columns.
    """
    df = glucose_data_verification(df)
    in_range_verification(in_range_interval)
    if time_units not in ['h', 'm', 's']:
        raise ValueError("time must be 'h', 'm' or 's'")
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
        't_br' if interval.right <= in_range_interval[0]
        else 't_ar' if interval.left >= in_range_interval[1]
        else 't_ir')
    )
    df_copy['ranges'] = map_ranges

    df_copy['time_diff'] = glucose_signals.apply(
            lambda signal: signal[column_name_timestamps].diff().dt.total_seconds()).values
    if time_units == 'h':
        df_copy['time_diff'] = df_copy['time_diff'] / 3600
    elif time_units == 'm':
        df_copy['time_diff'] = df_copy['time_diff'] / 60

    time_sum = df_copy.groupby([df_copy.index, 'ranges'], observed=False, sort=False)['time_diff'].sum()
    time_in_ranges_df = time_sum.unstack(level='ranges').fillna(0)

    for stat in ['t_ir', 't_ar', 't_br']:
        if stat not in time_in_ranges_df.columns:
            time_in_ranges_df[stat] = [0] * len(time_in_ranges_df)
    time_in_ranges_df['t_or'] = time_in_ranges_df['t_ar'] + time_in_ranges_df['t_br']

    return time_in_ranges_df


def percentage_time_in_ranges(df: pd.DataFrame, in_range_interval: list = [70, 180]) -> pd.DataFrame:
    """
    Calculates the percentage of time for specific ranges:
        - Percent time in range (pt_ir).
        - Percent time above range (pt_ar).
        - Percent time below range (pt_br).
        - Percent time out range (pt_or).

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
    percentage_time_in_ranges_df : pd.DataFrame
        A dataframe with ids of samples as index and percentage time in ranges as columns.
    """
    time_in_ranges_df = time_in_ranges(df, in_range_interval)

    percentage_time_in_ranges_df = pd.DataFrame()
    total = (time_in_ranges_df['t_ar'] + time_in_ranges_df['t_br'] + time_in_ranges_df['t_ir'])
    percentage_time_in_ranges_df['pt_ir'] = time_in_ranges_df['t_ir'] / total * 100
    percentage_time_in_ranges_df['pt_ar'] = time_in_ranges_df['t_ar'] / total * 100
    percentage_time_in_ranges_df['pt_br'] = time_in_ranges_df['t_br'] / total * 100
    percentage_time_in_ranges_df['pt_or'] = time_in_ranges_df['t_or'] / total * 100

    return percentage_time_in_ranges_df
