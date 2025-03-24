import argparse
import pandas as pd
from pathlib import Path
from glucostats.extract_statistics import ExtractGlucoStats
from glucostats.visualization.signal_visualization import plot_glucose_time_series
from glucostats.visualization.heatmaps import plot_interpatient_heatmap, plot_intrapatient_heatmap
from datetime import date

"""
          <a href="#">
            <img src="_images/logo_reduced.png" class="logo" alt="Logo">
          </a>
"""

def parse_arguments(parser):
    parser.add_argument('--n_cpus', default=0, type=int)
    parser.add_argument('--batching', default=1, type=int)

    return parser.parse_args()


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser(description='Main for nocturnal hypo')
    args = parse_arguments(args_parser)

    PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
    PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
    PATH_PROJECT_REPORTS_intra = Path.joinpath(PATH_PROJECT_DIR, 'reports','intra')
    PATH_PROJECT_REPORTS_inter = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'inter')
    PATH_PROJECT_REPORTS_cgm = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'cgm')
    PATH_PROJECT_REPORTS_cgm2= Path.joinpath(PATH_PROJECT_DIR, 'reports', 'cgm2')

    glucose_data_csv = pd.read_csv(Path.joinpath(PATH_PROJECT_DATA, 'glucose_data_in_days.csv'), index_col=0)
    glucose_data_csv['time'] = pd.to_datetime(glucose_data_csv['time'])
    glucose_data_csv_filtered = glucose_data_csv.groupby(level=0).filter(lambda x: len(x) > 1)
    glucose_data_csv_filtered = glucose_data_csv_filtered[glucose_data_csv_filtered.index != '11_2015-08-25']
    glucose_data_csv_filtered = glucose_data_csv_filtered[glucose_data_csv_filtered.index != '24_2015-08-31']
    glucose_data_csv_filtered = glucose_data_csv_filtered[glucose_data_csv_filtered.index != '31_2015-05-22']
    glucose_data_csv_filtered = glucose_data_csv_filtered[glucose_data_csv_filtered.index != '57_2015-07-05']

    signals_ids = glucose_data_csv_filtered.index.unique()
    glucose_data = glucose_data_csv_filtered.loc[signals_ids[0:1000]]

    list_statistics = ['percentage_time_in_ranges', 'distribution', 'grade_stats', 'qgc',
                       'variability']

    batch_size = args.batching
    n_cpus = args.n_cpus

    windowing = True
    windowing_method = 'number'
    windowing_param = 10
    windowing_start = 'head'
    windowing_overlap = False

    stats_extraction = ExtractGlucoStats(list_statistics, windowing, windowing_method, windowing_param,
                                         windowing_start, windowing_overlap, batch_size, n_cpus)

    stats_extraction = stats_extraction.configuration(threshold=100)
    statistics = stats_extraction.transform(glucose_data)
    time_ranges = stats_extraction.signals_time_ranges

    PATH_PROJECT_REPORTS_intra = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'intra_mag')
    plot_intrapatient_heatmap(statistics, time_ranges, '2', 'mean', [date(2015, 5, 23), date(2015, 6, 10)], saving_path = PATH_PROJECT_REPORTS_intra)



    plot_interpatient_heatmap(statistics, time_ranges, ['2', '3', '5', '7'], 'mean', [date(2015, 5, 23), date(2015, 5, 30)], saving_path = PATH_PROJECT_REPORTS_inter)
    plot_interpatient_heatmap(statistics, time_ranges, ['2', '3', '5', '7'], 'mean',
                              [date(2015, 5, 23), date(2015, 5, 30)], saving_path=PATH_PROJECT_REPORTS_inter)
    plot_glucose_time_series(glucose_data, ['2_2015-05-23','3_2015-05-23','5_2015-05-23','7_2015-05-23'], hyper_threshold=180, saving_path=PATH_PROJECT_REPORTS_cgm)
    plot_glucose_time_series(glucose_data, ['2_2015-05-23'], hyper_threshold=180, saving_path=PATH_PROJECT_REPORTS_cgm2)
