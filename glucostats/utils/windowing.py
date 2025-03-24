import pandas as pd
from datetime import timedelta
from glucostats.utils.format_verification import glucose_data_verification, windows_params_verification


def calculate_division_timestamps(df_signals: pd.DataFrame, windowing_method: str, windowing_param,
                                  windowing_start: str) -> pd.DataFrame:
    """
    Calculates the timestamps where the time series are going to be cutted in order to divide it into windows. Each
    signal from df_signals is processed independently and different timestamps can appear for the different signals
    if they are not embedded on the same time duration.

    Parameters
    ----------
    **df_signals : pd.DataFrame MultiIndex**

        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    **windowing_method : 'number', 'static', 'dynamic' or 'personalized'**

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

    **windowing_param :**

        The windowing param will be different depending on the windowing method.

        * Integer when method='number': number of windows desired.
        * List when method='static': window size, the first element is the days, the second element is the hours, the
          third element is the minutes and the forth element is the seconds.
        * List of lists when method='dynamic': window sizes, list of lists with the same format as when method='static'.
        * List of timestamps when method='personalized': timestamps that define where to cut the signal.

        See getting started for more details.

    **windowing_start : str**

        Where the window ranges begin to be calculated.

        * 'head': if window ranges begin to be calculated from the beginning of the signal.
        * 'tail': if window ranges begin to be calculated from the end of the signal.

        See getting started for more details.

    Return
    ------
    division_timestamps : pd.DataFrame
        A pd.DataFrame with unique ids of the signals as index and one column with the timestamps where the time series
        are going to be cutted in order to divide it into windows.
    """
    df_signals = glucose_data_verification(df_signals)
    windows_params_verification(windowing_method=windowing_method, windowing_param=windowing_param,
                                windowing_start=windowing_start)
    column_name_timestamps, column_name_glucose = df_signals.columns

    times_windows = []
    division_timestamps = pd.DataFrame()
    for unique_id, time_series in df_signals.groupby(level=0):
        first_date = time_series[column_name_timestamps].min()
        last_date = time_series[column_name_timestamps].max()
        total_duration = last_date - first_date

        if windowing_method == 'personalized':
            times_windows = [first_date]
            for timestamp in windowing_param:
                times_windows.append(timestamp)
            times_windows.append(last_date)

        elif windowing_method in ['number', 'static', 'dynamic']:
            window_lengths = []

            if windowing_method == 'number':
                n_windows = windowing_param
                window_length = total_duration / n_windows
                window_lengths = [window_length] * n_windows

            elif windowing_method == 'static':
                window_length = timedelta(days=windowing_param[0], hours=windowing_param[1],
                                          minutes=windowing_param[2], seconds=windowing_param[3])
                n_windows = total_duration // window_length
                window_lengths = [window_length] * n_windows

            elif windowing_method == 'dynamic':
                for time_ in windowing_param:
                    window_lengths.append(timedelta(days=time_[0], hours=time_[1],
                                                    minutes=time_[2], seconds=time_[3]))

            if windowing_start == 'tail':
                times_windows = [last_date]
                time_window_start = last_date
                for length in window_lengths:
                    time_window_start -= length
                    if time_window_start >= first_date:
                        times_windows.append(time_window_start)

            elif windowing_start == 'head':
                times_windows = [first_date]
                time_window_end = first_date
                for length in window_lengths:
                    time_window_end += length
                    if time_window_end <= last_date:
                        times_windows.append(time_window_end)

        times_windows = sorted(times_windows)

        division_timestamps = pd.concat([division_timestamps,
                                        pd.DataFrame([[times_windows]], index=[unique_id], columns=['time_windows'])])

    return division_timestamps


def create_windows(df_signals: pd.DataFrame, division_timestamps, windowing_start: str,
                   windowing_overlap: bool) -> (pd.DataFrame, pd.DataFrame):
    """
    Divide the signals into windows taking into account the division timestamps given.

    Parameters
    ----------
    df_signals : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    division_timestamps : pd.DataFrame
        Output of the calculate_division_timestamps function.

    **windowing_start : str**

        Where the window ranges begin to be calculated.

        * 'head': if window ranges begin to be calculated from the beginning of the signal.
        * 'tail': if window ranges begin to be calculated from the end of the signal.

        See getting started for more details.

    windowing_overlap : bool
        Whether the window ranges overlap with each other to create overlapping windows (True) or not (false).

    Return
    ------
    windowed_signals : pd.DataFrame
        A pd.DataFrame with the same format as df_signals but with signals divided into windows.

    signals_start_and_end: pd.DataFrame
        A pd.DataFrame where the index is the unique identifier of the signals or windows of the signals and the
        columns are the start and end timestamps of the signals or the windows of the signals.
    """
    df_signals = glucose_data_verification(df_signals)
    windows_params_verification(windowing_start=windowing_start, windowing_overlap=windowing_overlap)
    column_name_timestamps, column_name_glucose = df_signals.columns

    signals_start_and_end = pd.DataFrame()
    windowed_signals = pd.DataFrame()
    for unique_id, time_series in df_signals.groupby(level=0):
        windows = division_timestamps.loc[unique_id].item()
        for n_window in range(len(windows)-1):
            window_id = f"{unique_id}|{n_window}"
            start, end = windows[n_window], windows[n_window+1]

            if windowing_start == 'tail':
                if windowing_overlap:
                    filtered_df = time_series[time_series[column_name_timestamps] > start]
                    signals_start_and_end = pd.concat([signals_start_and_end,
                                                       pd.DataFrame({'start': start, 'end': windows[-1]},
                                                                    index=[window_id])])
                else:
                    filtered_df = time_series[(time_series[column_name_timestamps] > start) &
                                              (time_series[column_name_timestamps] <= end)]
                    signals_start_and_end = pd.concat([signals_start_and_end,
                                                       pd.DataFrame({'start': start, 'end': end},
                                                                    index=[window_id])])
            elif windowing_start == 'head':
                if windowing_overlap:
                    filtered_df = time_series[time_series[column_name_timestamps] < end]
                    signals_start_and_end = pd.concat([signals_start_and_end,
                                                       pd.DataFrame({'start': windows[0], 'end': end},
                                                                    index=[window_id])])
                else:
                    filtered_df = time_series[(time_series[column_name_timestamps] >= start) &
                                              (time_series[column_name_timestamps] < end)]
                    signals_start_and_end = pd.concat([signals_start_and_end,
                                                       pd.DataFrame({'start': start, 'end': end},
                                                                    index=[window_id])])
            else:
                raise ValueError('windowing_start must be either "tail" or "head".')

            filtered_df.index = [window_id] * len(filtered_df)
            windowed_signals = pd.concat([windowed_signals, filtered_df])

    return windowed_signals, signals_start_and_end
