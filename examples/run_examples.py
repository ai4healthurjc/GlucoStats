import os
import sys
from datetime import date
import numpy as np
import pandas as pd
from pathlib import Path

# Set path for examples folder and define paths
PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_PROJECT_EXAMPLES = Path.joinpath(PATH_PROJECT_DIR, 'examples')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glucostats.extract_statistics import ExtractGlucoStats
from glucostats.visualization.signal_visualization import plot_glucose_time_series
from glucostats.visualization.heatmaps import plot_intrapatient_heatmap


def load_cgm_example():

    df_data = pd.read_csv(
        os.path.join(PATH_PROJECT_EXAMPLES, f'cgm_users_examples.csv')
    )

    df_data.index = df_data['id']

    df_data_filtered = df_data[['time', 'glucose']]
    df_data_filtered = df_data_filtered.rename(columns={"time": "time", "glucose": "cgm"})
    df_data_filtered["time"] = pd.to_datetime(df_data_filtered["time"], errors="coerce")

    return df_data_filtered


df_cgm = load_cgm_example()
print(df_cgm)

# Plot glucose signals with ids: '2_2015-05-23', '3_2015-05-23', '5_2015-05-23', '8_2015-05-23'
plot_glucose_time_series(
    df_cgm,
    ['2_2015-05-23', '3_2015-05-23', '5_2015-05-23', '8_2015-05-23']
)

# Define list of statistics to compute
list_statistics = ['hypo_index', 'max_lbgi', 'mean']

# Define parameters for creating windows
windowing = True
windowing_method = 'number'
windowing_param = 4
windowing_start = 'tail'
windowing_overlap = False

# Instantiate ExtractGlucoStats class
stats_extraction = ExtractGlucoStats(
    list_statistics,
    windowing,
    windowing_method,
    windowing_param,
    windowing_start,
    windowing_overlap,
    batch_size=20,
    n_workers=4
)

# New configuration of intervals for hypoglycemia, euglycemia, hyperglycemia
in_range_interval = [70, 180]

# Class configuration and statistics extraction
stats_extraction.configuration(in_range_interval=in_range_interval)
df_cgm_stats = stats_extraction.transform(df_cgm)
df_cgm_stats = df_cgm_stats.dropna(thresh=len(df_cgm_stats), axis=1)
time_ranges = stats_extraction.signals_time_ranges

# Define heatmap parameters for visualization of a single patient (intra-patient visualization)
patient = '2'
days = [date(2015, 5, 23), date(2015, 5, 31)]

# Plot heatmap for mean statistic
selected_stat = 'mean'
plot_intrapatient_heatmap(df_cgm_stats, time_ranges, patient, selected_stat, days)

# Plot heatmap for max_lbgi
selected_stat = 'max_lbgi'
plot_intrapatient_heatmap(df_cgm_stats, time_ranges, patient, selected_stat, days)

# Plot heatmap for hypo_index
selected_stat = 'hypo_index'
plot_intrapatient_heatmap(df_cgm_stats, time_ranges, patient, selected_stat, days)