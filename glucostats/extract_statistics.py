import time
import pandas as pd
from colorlog import ColoredFormatter
import logging
from tqdm import tqdm
from multiprocessing import Pool
from typing import Tuple

from glucostats.utils.format_verification import list_statistics_verification, windows_params_verification
from glucostats.utils.batching import batching
from glucostats.utils.windowing import calculate_division_timestamps, create_windows
import glucostats.utils.constants as constants
from glucostats.stats import (time_stats, observations_stats, descriptive_stats, risks_stats, variability_stats,
                              control_stats)

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

# Definir el formato con colores
formatter = ColoredFormatter(
    "%(log_color)s%(levelname)s:%(reset)s %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ExtractGlucoStats(BaseEstimator, TransformerMixin):
    """
    Class that collects all the functionalities of the library and facilitates their use.

    Parameters
    ----------
    list_statistics : list
        A list containing the names of the statistics, subgroups of statistics or groups  of statistics to extract. See
        more details in getting started section.


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

    batch_size: int, default None
        If None, no batching is done. If is an integer, it will divide the dataset into batches of size equal
        to batch_size. If the number of signals is not multiple of batch_size, the last batch will contain fewer
        signals.

    n_workers: int, default 0
        Number of cpus to use for multiprocessing. Enables distributed processing by parllalel computation.
    """
    def __init__(self, list_statistics: list, windowing: bool = False, windowing_method: str = 'number',
                 windowing_param=4,  windowing_start: str = 'tail', windowing_overlap: bool = False,
                 batch_size: int = None, n_workers: int = 0):

        self.list_statistics = list_statistics_verification(list_statistics)

        windows_params_verification(windowing, windowing_method, windowing_param, windowing_start, windowing_overlap)
        self.windowing = windowing
        self.windowing_method = windowing_method
        self.windowing_param = windowing_param
        self.windowing_start = windowing_start
        self.windowing_overlap = windowing_overlap

        if not isinstance(batch_size, int) and batch_size is not None:
            raise ValueError('batch_size must be an integer or None. If None, no batching is done.')
        if isinstance(batch_size, int):
            if batch_size < 1:
                raise ValueError('batch_size must be greater than or equal to 1.')
        self.batch_size = batch_size

        if not isinstance(n_workers, int):
            raise ValueError('n_works must be an integer corresponding to the number of cpus for parallel computation.')
        if isinstance(n_workers, int):
            if n_workers < 0:
                raise ValueError('n_workers must be positive integer.')
        self.n_workers = n_workers

        self.stats_computed = False
        self.signals_time_ranges = None
        self.statistics = None

        self.stats_configuration = {
            'in_range_interval': [70, 180],
            'time_units': 'm',
            'ddof': 1,
            'quartiles': [0.25, 0.5, 0.75],
            'threshold': 0,
            'where': 'above',
            'a': 1.1,
            'b': 2.0,
            'c': 30,
            'd': 30,
            'ideal_bg': 120
        }

        self.data = None

    def configuration(self, in_range_interval: list = [70, 180], time_units: str = 'm', ddof: int = 1,
                      quartiles: list = [0.25, 0.5, 0.75], threshold: int or float = 0, where: str = 'above',
                      a: float = 1.1, b: float = 2.0, c: float = 30, d: float = 30, ideal_bg: int or float = 120):
        """
        Change configuration parameters for glucose statistics.

        Parameters
        ----------
        in_range_interval : list of int|float, default [70, 180]
            Interval defining whether glucose levels are within range or not. This parameter is useful for every
            statistic dependent of the ranges as time in ranges, observations in ranges or mean in ranges and for the
            calculation of glucose control indexes. The parameter must be a list with the lower and upper limit, default
            is [70, 180] in mg/dL.

        time_units : str, default 'm'
            Time units to calculate time in range. Can be 'h' (hours), 'm' (minutes) or 's' (seconds). Default 'm'.

        ddof : int, default 1
            Delta Degrees of Freedom corresponding to the adjustment made for the estimation of the mean in the sample
            data. The divisor used in calculations is N - ddof, where N represents the number of elements. Default 1.

        quartiles : list, default [0.25, 0.5, 0.75]
            List of values between 0 and 1 for the desired quantiles. Default [0.25, 0.5, 0.75].

        threshold : int | float, default 0
            The reference value from which the AUC will be calculated. Default 0.

        where : str, default 'above'
            The parameter that specifies the AUC will be calculated above the threshold or below the threshold. Must be
            'above' if above the threshold or 'below' if below the threshold, default 'above'.

        a: int | float, default 1.1
            Exponent, generally in the range from 1.0 to 2.0. Use in Hyperglycemic Index. Default value of 1.1.

        b: int | float, default 2.0
            Exponent, generally in the range from 1.0 to 2.0. Use in Hypoglycemic Index. Default value of 2.0.

        c, d : int | float, default 30
            Scaling factor. Default 30 to display Hyperglycemia Index, Hypoglycemia Index, and IGC on approximately the
            same numerical range as measurements of HBGI, LBGI and GRADE.

        ideal_bg : int | float, default=120
            The glucose level ideal value. Use for m-value calculation. Default 120.
        """
        self.stats_configuration = {
            'in_range_interval': in_range_interval,
            'time_units': time_units,
            'ddof': ddof,
            'quartiles': quartiles,
            'threshold': threshold,
            'where': where,
            'a': a,
            'b': b,
            'c': c,
            'd': d,
            'ideal_bg': ideal_bg
        }

        return self

    def statistics_computation(self, batch: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Windowing and statistics in list_statistics extraction from a batch of signals.

        Parameters
        ----------
        batch : pd.DataFrame MultiIndex
            A pd.DataFrame MultiIndex, where the first level (level 0) of the index is the unique identifier of the
            signals, the second level (level 1) of the index is the timestamps of each sample of the signals and with
            just one column containing the glucose levels values. Can be a piece of df_signals or the complete
            df_signals.

        Return
        ------
        stats : pd.DataFrame
            A pd.DataFrame where the index is the unique identifier of the signals or windows of the signals and the
            columns are the statistics extracted from the batch.

        signals_start_and_end: pd.DataFrame
            A pd.DataFrame where the index is the unique identifier of the signals or windows of the signals and the
            columns are the start and end timestamps of the signals or the windows of the signals.
        """
        if self.windowing:
            division_timestamps = calculate_division_timestamps(batch, self.windowing_method, self.windowing_param,
                                                                self.windowing_start)
            batch, signals_start_and_end = create_windows(batch, division_timestamps, self.windowing_start,
                                                          self.windowing_overlap)
        else:
            signals_start_and_end = 'error' # TODO check best descriptor

        dict_functions = {
            'time_in_ranges': lambda params: time_stats.time_in_ranges(batch, params['in_range_interval'], params['time_units']),
            'percentage_time_in_ranges': lambda params: time_stats.percentage_time_in_ranges(batch, params['in_range_interval']),
            'observations_in_ranges': lambda params: observations_stats.observations_in_ranges(batch, params['in_range_interval']),
            'percentage_observations_in_ranges': lambda params: observations_stats.percentage_observations_in_ranges(batch, params['in_range_interval']),
            'mean_in_ranges': lambda params: descriptive_stats.mean_in_ranges(batch, params['in_range_interval']),
            'distribution': lambda params: descriptive_stats.distribution(batch, params['ddof'], params['quartiles']),
            'complexity': lambda params: descriptive_stats.complexity(batch),
            'auc': lambda params: descriptive_stats.auc(batch, threshold=params['threshold'], where=params['where']),
            'g_indexes': lambda params: risks_stats.glucose_indexes(batch),
            'g_risks': lambda params: risks_stats.glycemia_risk(batch),
            'grade_stats': lambda params: risks_stats.grade(batch),
            'control_indexes': lambda params: control_stats.g_control(batch, params['in_range_interval'], params['a'], params['b'], params['c'], params['d']),
            'a1c': lambda params: control_stats.a1c_estimation(batch),
            'qgc': lambda params: control_stats.qgc_index(batch, params['ideal_bg']),
            'excursions': lambda params: variability_stats.signal_excursions(batch),
            'variability': lambda params: variability_stats.glucose_variability(batch),
        }

        stats = pd.DataFrame()
        for stat_name in self.list_statistics:
            if stat_name in constants.subgroups:
                stats = pd.concat([stats, dict_functions[stat_name](self.stats_configuration)], axis=1)
            else:
                for group, subgroups in constants.available_statistics.items():
                    for subgroup, subgroup_stats in subgroups.items():
                        if stat_name in subgroup_stats:
                            subgroup_of_stat_name = subgroup
                if stat_name == 'quartiles':
                    stat_name = list(map(lambda x: f'quartile_{x}', self.stats_configuration[stat_name]))
                stats = pd.concat([stats, dict_functions[subgroup_of_stat_name](self.stats_configuration)[stat_name]],
                                  axis=1)

        if self.windowing:
            stats.index = stats.index.str.split('|', expand=True)
            stats.index.names = ['unique_id', 'window']
            stats = stats.reset_index().set_index('unique_id')
            stats = stats.pivot_table(index=stats.index, columns='window', aggfunc='first')
            stats.columns = ['{}|{}'.format(col[0], col[1]) for col in stats.columns]

        return stats, signals_start_and_end

    def fit(self, X, y=None):
        """
        Function just created for making ExtractGlucoStats compatible in scikit-learn pipelines.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Main method for computing statistics encapsulating batching, multiprocessing, windowing and statistics
        extraction. Uses all funcionalities for computing statistics in list_statistics from signals in df_signals.

        Parameters
        ----------
        X : pd.DataFrame MultiIndex
            A pd.DataFrame MultiIndex, where the first level (level 0) of the index is the unique identifier of the
            signals, the second level (level 1) of the index is the timestamps of each sample of the signals and with
            just one column containing the glucose levels values.

        Return
        ------
        statistics_df : pd.DataFrame
            A pd.DataFrame where the index is the unique identifier of the signals and the columns are the
            statistics extracted from df_signals.
        """
        logger.info(f'Number of signals: {X.index.get_level_values(0).nunique()}.')
        logger.info(f'Number of samples: {X.shape[0]}.')

        self.data = X
        start = time.time()
        batches = batching(X, self.batch_size)
        end = time.time()
        print('Batching:', end - start)

        if self.n_workers > 0:
            logger.info(f'Distributed processing: cpus={self.n_workers}')
            start = time.time()
            with Pool(self.n_workers) as pool:
                results = list(tqdm(pool.imap(self.statistics_computation, batches),
                                    total=len(batches), desc="Statistics extraction", unit="batches", ncols=80))
            statistics_df = pd.concat([batch[0] for batch in results])
            signals_start_and_end = pd.concat([batch[1] for batch in results])
            end = time.time()
            logger.info('Extract statistics with cpus:'.format(end - start))
        else:
            logger.info(f'No distributed processing')
            statistics_df, signals_start_and_end = pd.DataFrame(), pd.DataFrame()
            start = time.time()
            for batch in tqdm(batches, desc="Statistics extraction", unit="batches", ncols=80):
                statistics, starts_and_ends = self.statistics_computation(batch)
                statistics_df = pd.concat([statistics_df, statistics])
                signals_start_and_end = pd.concat([signals_start_and_end, starts_and_ends])
            end = time.time()
            logger.info('Extract statistics with no cpus:', end - start)

        self.statistics = statistics_df
        self.signals_time_ranges = signals_start_and_end
        self.stats_computed = True

        return statistics_df

    def visualization(self, method, patients):
        pass
