import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from glucostats.utils.format_verification import glucose_data_verification

list_markers = ['o', '^', 'x', 's', 'd', '|', 'p', '>', '*', '<']


def plot_glucose_time_series(df_signals, signals_ids, hypo_1_threshold: int or float = 70,
                             hypo_2_threshold: int or float = 54, hyper_threshold: int or float = 180,
                             saving_path: str = None):
    """
    Generates a time series graph of glucose levels for a specific patient or for several patients, with colored
    backgrounds for type 1 hypoglycemia, type 2 hypoglycemia and hyperglycemia thresholds.

    Default ranges:

        - Hyperglycemia = glucose > hyper_threshold
        - Normal = hypo_1_threshold <= glucose <= hyper_threshold
        - Type 1 hypoglycemia = hypo_2_threshold <= glucose < hypo_1_threshold
        - Type 2 hypoglycemia = glucose < hypo_2_threshold

    Parameters
    ----------
    df_signals : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    signals_ids :
        Can an integer or string that refers to th identifier of a signal to graph or a list | array of the identifiers
        (integers or strings) of a number of signals to graph.

    hyper_threshold, hypo_2_threshold, hypo_1_threshold: int | float, default 180, 70, 54
        Thresholds for defining the different glucose ranges.

    saving_path : str | None, default None
        If path is provided, the resulting graph will be saved in the path. If None, no saving will be done just
        visualization.
    """
    for param in [hypo_1_threshold, hypo_2_threshold, hyper_threshold]:
        if not (isinstance(param, int) or isinstance(param, float)):
            raise ValueError(f'{param} must be int or float')
    if isinstance(signals_ids, int) or isinstance(signals_ids, str):
        signals_ids = [signals_ids]
    elif isinstance(signals_ids, list) or isinstance(signals_ids, np.ndarray):
        signals_ids = signals_ids
    else:
        raise TypeError(f'signals_ids must be int, string, list or array')
    if not (all(map(lambda x: isinstance(x, int) or isinstance(x, str), signals_ids))):
        raise ValueError(f'signals ids must be integers or strings')
    selected_signals = pd.DataFrame()
    for signal_id in signals_ids:
        signal = df_signals[df_signals.index == signal_id]
        if signal.empty:
            raise ValueError(f'{signal_id} is not in df_signals')
        else:
            selected_signals = pd.concat([selected_signals, signal])
    selected_signals = glucose_data_verification(selected_signals)

    column_name_timestamps, column_name_glucose = df_signals.columns

    max_value = selected_signals[column_name_glucose].max()
    first_date = selected_signals[column_name_timestamps].min()
    last_date = selected_signals[column_name_timestamps].max()

    plt.figure(figsize=(15, 7))
    for n, signal_id in enumerate(signals_ids):
        selected_signal = df_signals.loc[signal_id, :]
        plt.plot(selected_signal[column_name_timestamps], selected_signal[column_name_glucose],
                 label=f'Signal with id: {signal_id}', marker=list_markers[n], linestyle='-', markersize=1.5)

    plt.fill_between([first_date, last_date], 0, hypo_2_threshold, color='#FF6F61', alpha=0.4,
                     label='Type 2 hypoglycemia')
    plt.fill_between([first_date, last_date], hypo_2_threshold, hypo_1_threshold,
                     color='#FFB347', alpha=0.4, label='Type 1 hypoglycemia')
    plt.fill_between([first_date, last_date], hypo_1_threshold, hyper_threshold, color='#88B04B',
                     alpha=0.4, label='Normoglycemia')
    plt.fill_between([first_date, last_date], hyper_threshold, max_value, color='skyblue',
                     alpha=0.4, label='Hyperglycemia')

    plt.xlabel('Time',fontsize=18)
    plt.ylabel('Glucose (mg/dL)',fontsize=18)
    plt.title('Glucose Levels Over Time',fontsize=20)
    plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %B %Y %H:%M'))
    plt.xticks(rotation=60,fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if saving_path is not None:
        plt.savefig(saving_path)

    plt.show()
