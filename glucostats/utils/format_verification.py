import pandas as pd
import glucostats.utils.constants as constants
from typing import List
from datetime import datetime


def list_statistics_verification(list_statistics: list) -> List:
    """
    Function for verifying if list_statistics has the correct format and contains only names of available statistics,
    subgroups or groups. Also return list statistics with statistics names ordered to be more efficient and
    eliminate duplicates.

    PARAMS
    ------
    list_statistics : list
        A list containing the names of the statistics, subgroups of statistics or groups  of statistics to extract. See
        more details in getting started section.

    RETURN
    ------
    list_statistics_ordered : list
        A list containing the names of the statistics, subgroups of statistics or groups  of statistics to extract to
        be more efficient and in the correct format.
    """
    if not isinstance(list_statistics, list):
        raise TypeError("list_statistics must be a list.")
    if not all(map(lambda x: isinstance(x, str), list_statistics)):
        raise TypeError("list_statistics elements must be strings.")
    if len(list_statistics) == 0:
        raise ValueError("list_statistics must have at least one element.")

    for name in list_statistics:
        if name not in constants.possible_names:
            raise ValueError(f'"{name}" statistic, subgroup or group not available.')

    list_statistics_ordered = []

    groups_selected = list(set(list_statistics) & set(constants.groups))
    for group in groups_selected:
        del_subgroups = constants.available_statistics[group].keys()
        del_stats = sum(constants.available_statistics[group].values(), [])
        list_statistics = list(set(list_statistics) - set(del_subgroups) - set(del_stats))
        list_statistics_ordered += list(constants.available_statistics[group].keys())

    subgroups_selected = list(set(list_statistics) & set(constants.subgroups))
    for subgroup in subgroups_selected:
        for group in constants.available_statistics.values():
            if subgroup in group.keys():
                del_stats = group[subgroup]
                list_statistics = list(set(list_statistics) - set(del_stats))
    list_statistics_ordered += subgroups_selected

    stats_selected = list(set(list_statistics) & set(constants.statistics))
    list_statistics_ordered += stats_selected

    return list_statistics_ordered


def glucose_data_verification(df_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Function for verifying if df_signals has the correct format.

    PARAMS
    ------
    df_signals : pd.DataFrame
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    RETURN
    ------
    df_signals : pd.DataFrame
        Only return df_signals if it has the correct format.
    """

    if not isinstance(df_signals, pd.DataFrame):
        raise TypeError('df_signals must be a multi-indexed dataframe.')

    if df_signals.index.nlevels != 1:
        raise ValueError('df_signals must have only one index level with signals ids.')
    if not all(map(lambda x: isinstance(x, int) or isinstance(x, str), df_signals.index)):
        raise TypeError('Index must be integers or strings.')

    if df_signals.shape[1] != 2:
        raise ValueError('df_signals must have two columns: timestamps and glucose levels.')
    column_name_timestamps, column_name_glucose = df_signals.columns

    if not (all(map(lambda x: isinstance(x, datetime), df_signals[column_name_timestamps]))):
        raise ValueError('First column corresponding to timestamps contain non datetime type values.')

    signals_len = df_signals.groupby(level=0).apply(len)
    if len(signals_len[signals_len <= 1]) > 0:
        raise ValueError(f'Glucose signals must have more than one sample. {list(signals_len[signals_len <= 1].index)} '
                         f'signals have only one or no samples.')

    if not all(map(lambda x: (isinstance(x, float) or isinstance(x, int)), df_signals[column_name_glucose])):
        raise TypeError('Glucose levels must be integers or float values.')
    if not all(map(lambda x: x >= 0, df_signals[column_name_glucose])):
        raise ValueError('Glucose levels must be positive values.')

    return df_signals


def windows_params_verification(windowing: bool = False, windowing_method: str = 'number', windowing_param=4,
                                windowing_start: str = 'tail', windowing_overlap: bool = False):
    """
    Function for verifying if windowing parameters are in the correct format and type of variable and if they are
    coherent.

    Parameters
    ----------
    windowing: bool, default False
        Wether to divide signals into windows (True) or not (False).

    **windowing_method : 'number', 'static', 'dynamic' or 'personalized', default 'number'**

        The method chosen for signal windowing.

        * 'number': this method allows the signal to be divided into an indicated number of windows determined by the
          user.
        * 'static': this method allows the signal to be divided into windows of an indicated size determined by the
          user.
        * 'dynamic': this method allows the signal to be divided into windows of a different indicated sizes determined
          by the user.
        * 'personalized': this method allows the signal to be divided taking into account timestamps determined by
          the user.

        See getting started for more details.

    **windowing_param : default 4**

        The windowing param will be different depending on the windowing method.

        * Integer when method='number': number of windows desired.
        * List when method='static': window size, the first element is the days, the second element is the hours, the
          third element is the minutes and the forth element is the seconds.
        * List of lists when method='dynamic': window sizes, list of lists with the same format as when method='static'.
        * List of timestamps when method='personalized': timestamps that define where to cut the signal.

        See getting started for more details.

    **windowing_start : str, default 'tail'**

        Where the window ranges begin to be calculated.

        * 'head': if window ranges begin to be calculated from the beginning of the signal.
        * 'tail': if window ranges begin to be calculated from the end of the signal.

        See getting started for more details.

    windowing_overlap : bool, default False
        Whether the window ranges overlap with each other to create overlapping windows (True) or not (false).
    """

    if not isinstance(windowing, bool):
        raise TypeError('windowing must be a boolean: True if desired or False in contrast.')

    if not isinstance(windowing_method, str):
        raise TypeError('windowing_method must be a string.')
    if windowing_method not in constants.windowing_methods:
        raise ValueError(f'"{windowing_method}" method not available. Available: {constants.windowing_methods}')

    if windowing_method == 'number':
        if not isinstance(windowing_param, int):
            raise TypeError('window_param must be an integer for "number" method, corresponding to the number of '
                             'windows.')
        if windowing_param < 1:
            raise ValueError('window_param must be a positive integer greater than 0 for "number" method, corresponding'
                             ' to the number of windows.')
    elif windowing_method == 'static':
        if not isinstance(windowing_param, list):
            raise TypeError('window_param must be a list for "static" method, correspondig to window size.')
        if len(windowing_param) != 4:
            raise ValueError('window_param must be a list with format [days, hours, minutes, seconds] for "static" '
                             'method, correspondig to window size.')
        if not all(isinstance(element, int) and element >= 0 for element in windowing_param):
            raise TypeError('windowing_param must be a list with format [days, hours, minutes, seconds] of '
                             'positive integers for "static" method, correspondig to window size.')
    elif windowing_method == 'dynamic':
        if not isinstance(windowing_param, list):
            raise TypeError('window_param must be a list of lists for "dynamic" method, corresponding to '
                             'window sizes.')
        for time_ in windowing_param:
            if not isinstance(time_, list):
                raise TypeError('elements of window_param must be lists for "dynamic" method, correspondig to '
                                 'window size.')
            if len(time_) != 4:
                raise ValueError('elements of window_param must be lists with format [days, hours, minutes, seconds] '
                                 'for "dynamic" method, correspondig to window size.')
            if not all(isinstance(element, int) and element >= 0 for element in time_):
                raise TypeError('elements of windowing_param must be lists with format [days, hours, minutes, '
                                 'seconds] of positive integers for "dynamic" method, correspondig to window size.')
    else:
        if not isinstance(windowing_param, list):
            raise TypeError('window_param must be a list of timestamps for "personalized" method, '
                             'corresponding to where the signal is going to be cutted.')
        for timestamp in windowing_param:
            if not isinstance(timestamp, datetime):
                raise TypeError('elements of window_param must be timestamps of type datetime for "personalized" '
                                 'method, corresponding to where the signal is going to be cutted.')

    if not isinstance(windowing_start, str):
        raise TypeError('windowing_start must be a string.')
    if windowing_start != 'tail' and windowing_start != 'head':
        raise ValueError('windowing_start must be either "tail" or "head", corresponding to where ranges '
                         'begin being calculated.')

    if not isinstance(windowing_overlap, bool):
        raise TypeError('windowing_overlap must be a boolean: True if overlaping windows are desired of False '
                         'in contrast.')


def in_range_verification(in_range_interval):
    """
    Function for verifying if in_range_interval has the correct format and coherent limits of the range

    Parameters
    ----------
    in_range_interval : list of int|float
        Interval defining whether glucose levels are within range or not.
    """
    if not isinstance(in_range_interval, list):
        raise TypeError("in_range_interval must be a list")
    if not len(in_range_interval) == 2:
        raise ValueError("in_range_interval must be length of 2 numbers")
    if not all(map(lambda x: (isinstance(x, float) or isinstance(x, int)) and x >= 0, in_range_interval)):
        raise TypeError("in_range_interval must be a list of positive integer or float values corresponding to the "
                        "lower limit and the upper limit of in range interval.")
