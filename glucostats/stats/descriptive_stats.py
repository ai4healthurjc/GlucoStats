import numpy as np
import pandas as pd
import neurokit2 as nk
from neurokit2.complexity import fractal_dfa
from glucostats.utils.format_verification import glucose_data_verification, in_range_verification


def mean_in_ranges(df: pd.DataFrame, in_range_interval: list = [70, 180]) -> pd.DataFrame:
    """
    Calculates the mean glucose levels for specific ranges:
        - Mean glucose in range (mean_ir).
        - Mean glucose above range (mean_ar).
        - Mean glucose below range (mean_br).
        - Mean glucose out of range (mean_or).

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
    mean_in_ranges_df : pd.DataFrame
        A dataframe with ids of samples as index and mean glucose levels in ranges as columns.
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
        'mean_br' if interval.right <= in_range_interval[0]
        else 'mean_ar' if interval.left >= in_range_interval[1]
        else 'mean_ir')
    )
    df_copy['ranges'] = map_ranges

    or_values = df_copy[df_copy['ranges'].isin(['mean_ar', 'mean_br'])]
    mean_or = or_values.groupby(level=0, sort=False)[column_name_glucose].mean()

    mean_glucose = df_copy.groupby([df_copy.index, 'ranges'], observed=False, sort=False)[column_name_glucose].mean()
    mean_in_ranges_df = mean_glucose.unstack(level='ranges')

    for stat in ['mean_ir', 'mean_ar', 'mean_br']:
        if stat not in mean_in_ranges_df.columns:
            mean_in_ranges_df[stat] = [np.nan] * len(mean_in_ranges_df)
    mean_in_ranges_df['mean_or'] = mean_or

    return mean_in_ranges_df


def distribution(df: pd.DataFrame, ddof: int = 1, qs: list = [0.25, 0.5, 0.75]) -> pd.DataFrame:
    """
    Calculates the signal statistics from glucose data:
        - Maximum of glucose levels (max).
        - Minimum of glucose levels (min).
        - Maximum difference glucose levels (max_diff).
        - Mean of glucose levels (mean).
        - Standard deviation of glucose levels (std).
        - Quantiles (quantile_x).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    ddof : int, default 1
        Delta Degrees of Freedom corresponding to the adjustment made for the estimation of the mean in the sample
        data. The divisor used in calculations is N - ddof, where N represents the number of elements. Must be an
        integer, default 1.

    qs : list, default [0.25, 0.5, 0.75]
        List of values between 0 and 1 for the desired quantiles. Default [0.25, 0.5, 0.75].

    Returns
    -------
    distribution_df : pd.DataFrame
        A dataframe with ids of samples as index and signal statistics as columns.
    """
    df = glucose_data_verification(df)
    if ddof != 0 and ddof != 1:
        raise ValueError("ddof must be 0 or 1")
    if not isinstance(qs, list):
        raise TypeError("qs must be a list of the quartiles desired")
    if not all(map(lambda x: isinstance(x, float) and 0 <= x <= 1, qs)):
        raise ValueError("qs must be a list of float values between 0 and 1 corresponding the quartiles desired.")
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    glucose_signals = df_copy.groupby(level=0, sort=False)
    glucose_signals_values = glucose_signals[column_name_glucose]

    distribution_df = pd.DataFrame()
    distribution_df['max'] = glucose_signals_values.max()
    distribution_df['min'] = glucose_signals_values.min()
    distribution_df['max_diff'] = distribution_df['max'] - distribution_df['min']
    distribution_df['mean'] = glucose_signals_values.mean()
    distribution_df['std'] = glucose_signals_values.std(ddof=ddof)
    if len(qs) > 0:
        for q in qs:
            distribution_df[f'quartile_{q}'] = glucose_signals_values.quantile(q=q, interpolation='linear')
    if 0.25 in qs and 0.75 in qs:
        distribution_df['iqr'] = (distribution_df['quartile_0.75'] - distribution_df[f'quartile_0.25'])
    else:
        if 0.25 not in qs:
            distribution_df[f'quartile_0.25'] = glucose_signals_values.quantile(q=0.25, interpolation='linear')
        if 0.75 not in qs:
            distribution_df[f'quartile_0.75'] = glucose_signals_values.quantile(q=0.75, interpolation='linear')
        distribution_df['iqr'] = (distribution_df[f'quartile_0.75'] - distribution_df[f'quartile_0.25'])

    return distribution_df


def complexity(df: pd.DataFrame, scale='default', overlap: bool = True, integrate: bool = True, order: int = 1,
               show: bool = False, delay=1, dimension=2, tolerance='sd') -> pd.DataFrame:
    """
    Calculates the signal complexity:
        - Entropy (entropy).
        - Detrended Fluctuation Analysis (dfa).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    scale, overlap, integrate, order, show, delay, dimension, tolerance:
        Watch neurokit2 for more information

    Returns
    -------
    complexity_df : pandas.DataFrame
        A dataframe with ids of samples as index and signal complexity statistics as columns.
    """
    df = glucose_data_verification(df)
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    glucose_signals = df_copy.groupby(level=0, sort=False)

    def try_dfa(x):
        try:
            return fractal_dfa(x, scale=scale, overlap=overlap, integrate=integrate, order=order, show=show)[0]
        except:
            return np.nan

    def try_entropy(x, t_delay, dim, tol):
        try:
            if t_delay is None:
                t_delay, _ = nk.complexity_delay(x)
            if dim is None:
                dim, _ = nk.complexity_dimension(x, delay=t_delay)
            if tol is None:
                tol, _ = nk.complexity_tolerance(x, delay=t_delay, dimension=dim)
            entropy, _ = nk.entropy_sample(x, delay=t_delay, dimension=dim, tolerance=tol)
            return entropy
        except:
            return np.nan

    complexity_df = pd.DataFrame(index=df_copy.index.unique())
    complexity_df['dfa'] = glucose_signals[column_name_glucose].apply(try_dfa).values
    complexity_df['entropy'] = glucose_signals[column_name_glucose].apply(
        lambda x: try_entropy(x, t_delay=delay, dim=dimension, tol=tolerance)).values

    return complexity_df


def auc(df: pd.DataFrame, threshold: int or float = 0., where: str = 'above') -> pd.DataFrame:
    """
    Calculates the Area Under the Curve (AUC) using the trapezoidal rule:
        - Area Under the curve (auc).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    threshold : int | float, default 0
        The reference value from which the AUC will be calculated. Default 0.

    where : str, default 'above'
        If 'above', the AUC will be calculated above the threshold. If 'below', the AUC will be calculated below the
        threshold. Default 'above'.

    Returns
    -------
    auc_df : pandas.DataFrame
        A dataframe with ids of samples as index and auc value as column.
    """
    df = glucose_data_verification(df)
    if not (isinstance(threshold, float) or isinstance(threshold, int)):
        raise TypeError('threshold must be a positive integer or float')
    if threshold < 0:
        raise ValueError('threshold must be a positive integer or float')
    if where != 'above' and where != 'below':
        raise ValueError('where must be either "above" or "below"')
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    glucose_signals = df_copy.groupby(level=0, sort=False)

    df_copy['time_diff'] = glucose_signals.apply(
            lambda signal: signal['time'] - min(signal['time'])).values
    df_copy['time_diff'] = df_copy['time_diff'].apply(lambda x: x.total_seconds() / 3600)

    df_copy['glucose_diff'] = df_copy[column_name_glucose].apply(
        lambda x:
        (np.maximum(x, threshold) - threshold)
        if where == 'above' else
        (threshold - np.minimum(x, threshold))).values

    auc_df = pd.DataFrame()
    auc_df['auc'] = glucose_signals.apply(
        lambda x: np.trapz(y=x['glucose_diff'], x=x['time_diff']))

    return auc_df
