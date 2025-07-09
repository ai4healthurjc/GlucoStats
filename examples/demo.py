"""
Basic demo for the glucostats library

Loads the example dataset, computes some metrics, and shows a simple plot.
"""

import pandas as pd
from glucostats.datasets import load_glucodata
from glucostats.extract_statistics import ExtractGlucoStats
from glucostats.visualization.signal_visualization import plot_glucose_time_series
from glucostats.visualization.heatmaps import plot_intrapatient_heatmap


def main():
    # Load glucose data
    df_data = load_glucodata()
    print("Loaded data")

    # Change datetime format and set index for multiple signals
    df_data.index = df_data['id']
    df_data = df_data[['time', 'glucose']]
    df_data["time"] = pd.to_datetime(df_data["time"], errors="coerce")
    print(df_data.head())

    # Plot glucose signals with ids: '2_2015-05-23', '3_2015-05-23', '5_2015-05-23', '8_2015-05-23'
    plot_glucose_time_series(
        df_data,
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
    df_cgm_stats = stats_extraction.transform(df_data)
    df_cgm_stats = df_cgm_stats.dropna(thresh=len(df_cgm_stats), axis=1)
    time_ranges = stats_extraction.signals_time_ranges

    print(df_cgm_stats)


if __name__ == "__main__":
    main()
