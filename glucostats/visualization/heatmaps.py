import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from datetime import datetime, date, time, timedelta


def plot_intrapatient_heatmap(df_stats: pd.DataFrame, signals_time_ranges: pd.DataFrame,
                              patient: str, stat: str, days: list, saving_path: str = None):
    """
    Generates an intrapatient heatmap. See the definition in visualization section in getting started.

    Parameters
    ----------
    df_stats : pd.DataFrame
        A pd.DataFrame where the index is the unique identifier of the signals and the columns are the
        statistics extracted from the df_signals. Can be obtained calling the attribute .statistics of
        ExtractGlucoStats.

    signals_time_ranges: pd.DataFrame
        A pd.DataFrame where the index is the unique identifier of the signals or windows of the signals and the
        columns are the start and end timestamps of the signals or the windows of the signals. Can be obtained calling
        the attribute .signals_time_ranges of ExtractGlucoStats.

    patient: str
        The id of the patient to analyze.

    stat : str
        The name of the statistic to be plotted.

    days: list
        A list with just two elements, the fist element will be the start date and the second element will be the end
        date of the period to analyze. A maximum of 31 days can be analyzed at once.

    saving_path : str | None, default None
        If path is provided, the resulting graph will be saved in the path. If None, no saving will be done just
        visualization.
    """
    if not isinstance(df_stats, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame.')

    if not isinstance(stat, str):
        raise TypeError('stat must be a string.')
    if len(df_stats.columns[0].split('|')) > 1:
        df_stats = df_stats.filter(regex=rf'^{stat}\|')
    else:
        df_stats = df_stats.filter(items=[f'{stat}'], axis=1)
    if df_stats.empty:
        raise ValueError(f'stat {stat} is not in df_stats.')

    if not isinstance(patient, str):
        raise TypeError('patient must be a string corresponding to the id of the patient to analyze.')
    patient_stats = df_stats.filter(regex=rf'^{patient}\_', axis=0)
    if len(patient_stats) == 0:
        raise ValueError(f'patient {patient} must is not in in df_stats.')

    if not isinstance(days, list):
        raise TypeError('days must be a list.')
    if len(days) != 2:
        raise ValueError('days must be a list with two elements.')
    if not all(map(lambda x: isinstance(x, date), days)):
        raise TypeError('elements of days must be datetime objects.')
    num_days = days[1] - days[0]
    if num_days > timedelta(days=31):
        raise ValueError(f'A maximum of 31 days can be represented.')

    column_names = [days[0]]
    for day in range(1, num_days.days+1):
        column_names.append(days[0]+day*timedelta(days=1))
    all_days = dict.fromkeys(column_names)
    signals_starts_and_ends = signals_time_ranges.filter(regex=rf'^{patient}\_', axis=0)
    for signal_id in patient_stats.index:
        if len(df_stats.columns[0].split('|')) > 1:
            signal_starts_and_ends = signals_starts_and_ends[signals_starts_and_ends.index.str.startswith(f"{signal_id}|")]
        else:
            signal_starts_and_ends = signals_starts_and_ends[signals_starts_and_ends.index.str == f"{signal_id}"]
        datetimes_in_singal = pd.concat([signal_starts_and_ends['start'], signal_starts_and_ends['end']])
        day_of_signal = datetimes_in_singal.iloc[0].date()
        day_start = datetime.combine(day_of_signal, time(0, 0, 0))
        day_end = day_start + timedelta(days=1)
        if not all(map(lambda x: day_start <= x <= day_end, datetimes_in_singal)):
            raise ValueError(f'signal {signal_id} is not embedded in a day.')
        else:
            if days[0] <= day_of_signal <= days[1]:
                all_days.update({day_of_signal: signal_id})

    signals_ids_between_days = list(filter(lambda x: x is not None, all_days.values()))
    if len(signals_ids_between_days) == 0:
        raise ValueError(f'No signals between the days given.')
    signals_between_days = patient_stats[patient_stats.index.isin(signals_ids_between_days)]

    signals_between_days_information = {}
    for signal_id, signal_stats in signals_between_days.iterrows():
        signal_windows_and_values = []
        n_windows = len(signal_stats)
        if n_windows > 1:
            for window in range(n_windows):
                start = signals_starts_and_ends.loc[f'{signal_id}|{window}']['start']
                end = signals_starts_and_ends.loc[f'{signal_id}|{window}']['end']
                window_size = (end - start).total_seconds() / 3600

                custom_date = datetime(1, 1, 1)
                start = datetime.combine(custom_date, start.time())
                end = start + timedelta(hours=window_size)
                value = signal_stats[f'{stat}|{window}']
                signal_windows_and_values.append((start, end, window_size, value))
        else:
            start = signals_starts_and_ends.loc[f'{signal_id}']['start']
            end = signals_starts_and_ends.loc[f'{signal_id}']['end']
            signal_size = (end - start).total_seconds() / 3600

            custom_date = datetime(1, 1, 1)
            start = datetime.combine(custom_date, start.time())
            end = start + timedelta(hours=signal_size)
            value = signal_stats[f'{stat}']
            signal_windows_and_values.append((start, end, signal_size, value))

        signals_between_days_information.update({signal_id: signal_windows_and_values})

    heatmap = {day: signals_between_days_information[signal]
               if signal is not None else
               None for day, signal in all_days.items()}
    title = (f"Heatmap for {stat} statistic for patient {patient} between the "
             f"days {days[0].strftime('%Y/%m/%d')}-{days[1].strftime('%Y/%m/%d')}")

    colormaps = [
        'viridis', 'plasma', 'GnBu',
        'RdYlBu', 'coolwarm', 'Spectral_r'
    ]

    for i in colormaps:

        fig, ax = plt.subplots(figsize=(15, 10))

        list_values = [value for sublist in heatmap.values() if sublist is not None for value in sublist]
        max_value = max(list_values, key=lambda x: x[3])[3]
        min_value = min(list_values, key=lambda x: x[3])[3]

        cmap = plt.get_cmap(i)
        norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

        for n, (column_name, column_values) in enumerate(heatmap.items()):
            if column_values is not None:
                for (start, end, duration, value) in column_values:
                    color = cmap(norm(value))
                    ax.bar(n, duration, bottom=start, width=1, color=color, edgecolor=None)
                    ax.bar(n, duration, bottom=end, width=1, color='white', edgecolor=None)

        ax.set_xlabel('Dates', fontsize=20)
        ax.set_ylabel('Time', fontsize=20)
        ax.set_title(title, fontsize=22)

        ax.set_xticks(list(range(len(heatmap.keys()))))
        ax.set_xticklabels(heatmap.keys())

        ax.yaxis_date()
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%H-%M-%S'))

        # ax.set_ylim(max_time, min_time)
        ax.set_ylim(datetime(1, 1, 1, 23, 59, 59),
                    datetime(1, 1, 1, 0, 0, 0))
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=20)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'{stat} value', fontsize=22)
        cbar.ax.tick_params(labelsize=20)

        plt.xticks(rotation=90)
        plt.tight_layout()

        if saving_path is not None:
            plt.savefig(saving_path, format="pdf", bbox_inches="tight")

        plt.show()


def plot_interpatient_heatmap(df_stats: pd.DataFrame, signals_time_ranges: pd.DataFrame,
                              patients: list, stat: str, days: list, saving_path: str = None):
    """
    Generates an interpatient heatmap. See the definition in visualization section in getting started.

    Parameters
    ----------
    df_stats : pd.DataFrame
        A pd.DataFrame where the index is the unique identifier of the signals and the columns are the
        statistics extracted from the df_signals. Can be obtained calling the attribute .statistics of
        ExtractGlucoStats.

    signals_time_ranges: pd.DataFrame
        A pd.DataFrame where the index is the unique identifier of the signals or windows of the signals and the
        columns are the start and end timestamps of the signals or the windows of the signals. Can be obtained calling
        the attribute .signals_time_ranges of ExtractGlucoStats.

    patients: list
        The list of the ids of the patients to analyze.

    stat : str
        The name of the statistic to be plotted.

    days: list
        A list with just two elements, the fist element will be the start date and the second element will be the end
        date of the period to analyze. A maximum of 31 days can be analyzed at once.

    saving_path : str | None, default None
        If path is provided, the resulting graph will be saved in the path. If None, no saving will be done just
        visualization.
    """
    if not isinstance(df_stats, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame.')

    if not isinstance(stat, str):
        raise TypeError('stat must be a string.')
    if len(df_stats.columns[0].split('|')) > 1:
        df_stats = df_stats.filter(regex=rf'^{stat}\|')
    else:
        df_stats = df_stats.filter(items=[f'{stat}'], axis=1)
    if df_stats.empty:
        raise ValueError(f'stat {stat} is not in df_stats.')

    if not isinstance(patients, list):
        raise TypeError('patients must be a list of patients ids.')
    if not all(map(lambda x: isinstance(x, str) or isinstance(x, int), patients)):
        raise TypeError('patients ids must in patients mut be strings or integers.')
    patients_stats = pd.DataFrame()
    for patient in patients:
        patient_stats = df_stats.filter(regex=rf'^{patient}\_', axis=0)
        if len(patient_stats) == 0:
            raise ValueError(f'patient {patient} must is not in in df_stats.')
        patients_stats = pd.concat([patients_stats, patient_stats])

    if not isinstance(days, list):
        raise TypeError('days must be a list.')
    if len(days) != 2:
        raise ValueError('days must be a list with two elements.')
    if not all(map(lambda x: isinstance(x, date), days)):
        raise TypeError('elements of days must be datetime objects.')
    num_days = days[1] - days[0]
    if num_days > timedelta(days=31):
        raise ValueError(f'A maximum of 31 days can be represented.')

    signals_starts_and_ends = signals_time_ranges.filter(regex=rf'^({"|".join(patients)})\_', axis=0)
    signals_between_days_starts_and_ends = signals_starts_and_ends[
        (signals_starts_and_ends['start'].apply(lambda x: x.date()) <= days[1]) &
        (signals_starts_and_ends['end'].apply(lambda x: x.date()) >= days[0])
    ]
    if signals_between_days_starts_and_ends.empty:
        raise ValueError(f'No signals between the days given.')

    heatmap = {}
    for patient in patients:
        patient_between_days_information = []
        patient_stats = patients_stats.filter(regex=rf'^{patient}\_', axis=0)
        patient_signals_between_days_starts_and_ends = signals_between_days_starts_and_ends.filter(regex=rf'^{patient}\_', axis=0)

        for window_id, (start, end) in patient_signals_between_days_starts_and_ends.iterrows():
            if patients_stats.shape[1] > 1:
                signal_id = window_id.split('|')[0]
                stat_name = f"{stat}|{window_id.split('|')[1]}"
            else:
                signal_id = window_id
                stat_name = stat
            window_duration = (end - start).total_seconds() / 3600
            value = patient_stats.loc[signal_id][stat_name]
            patient_between_days_information.append((start, end, window_duration, value))

        if len(patient_between_days_information) == 0:
            heatmap.update({patient: None})
        else:
            heatmap.update({patient: patient_between_days_information})

    title = (f"Heatmap for {stat} statistic for {len(patients)} patients between the "
             f"days {days[0].strftime('%Y/%m/%d')}-{days[1].strftime('%Y/%m/%d')}")

    colormaps = [
        'viridis', 'plasma', 'YlGnBu',
        'RdYlBu', 'coolwarm', 'Spectral_r'
    ]

    for i in colormaps:

        fig, ax = plt.subplots(figsize=(15, 10))

        list_values = [value for sublist in heatmap.values() if sublist is not None for value in sublist]
        max_value = max(list_values, key=lambda x: x[3])[3]
        min_value = min(list_values, key=lambda x: x[3])[3]

        cmap = plt.get_cmap(i)
        norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

        for n, (row_name, row_values) in enumerate(heatmap.items()):
            if row_values is not None:
                for (start, end, duration, value) in row_values:
                    color = cmap(norm(value))
                    ax.barh(n, duration, left=start, height=1, color=color, edgecolor=None)
                    ax.barh(n, duration, left=end, height=1, color='white', edgecolor=None)

        ax.set_xlabel('Dates and time', fontsize=22)
        ax.set_ylabel('Patients ids', fontsize=22)
        ax.set_title(title, fontsize=22)

        ax.set_yticks(list(range(len(heatmap.keys()))))
        ax.set_yticklabels(heatmap.keys())

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H:%M:%S'))

        ax.set_xlim(datetime.combine(days[0], time(0, 0, 0)),
                    datetime.combine(days[1], time(23, 59, 59)))

        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=20)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'{stat} value', fontsize=22)
        cbar.ax.tick_params(labelsize=20)

        plt.xticks(rotation=90)
        plt.tight_layout()

        if saving_path is not None:
            plt.savefig(saving_path, format="pdf", bbox_inches="tight")

        plt.show()
