import pandas as pd
import numpy as np
from glucostats.utils.format_verification import glucose_data_verification


def glucose_indexes(df: pd.DataFrame):
    """
    Calculates glucose indexes:
        - Low Blood Glucose Index (lbgi).
        - High Blood Glucose Index (hbgi).
        - Maximum lbgi (max_lbgi).
        - Maximum hbgi (max_hbgi).
        - Blood glucose risk index (bgri).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    Returns
    -------
    gi_df : pandas.DataFrame
        A Dataframe with ids of samples as index and glucose indexes as columns.
    """
    df = glucose_data_verification(df)
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    glucose_signals = df_copy.groupby(level=0, sort=False)

    df_copy['risk'] = df_copy[column_name_glucose].apply(lambda x: ((np.log(x) ** 1.084) - 5.381) * 1.509)
    df_copy['risk_l'] = df_copy['risk'].apply(lambda x: 0 if x >= 0 else 10 * (x ** 2))
    df_copy['risk_h'] = df_copy['risk'].apply(lambda x: 0 if x <= 0 else 10 * (x ** 2))

    gi_df = pd.DataFrame()
    gi_df['lbgi'] = glucose_signals['risk_l'].mean()
    gi_df['max_lbgi'] = glucose_signals['risk_l'].max()
    gi_df['hbgi'] = glucose_signals['risk_h'].mean()
    gi_df['max_hbgi'] = glucose_signals['risk_h'].max()
    gi_df['bgri'] = gi_df['lbgi'] + gi_df['hbgi']

    return gi_df


def glycemia_risk(df: pd.DataFrame):
    """
    Calculates glucose indexes:
        - Percentage time of very low glucose (vlow).
        - Percentage time of low glucose (low).
        - Percentage time of high glucose (high).
        - Percentage time of very high glucose (vhigh).
        - Glycemia risk index (gri).

    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    Returns
    -------
    gr_df : pandas.DataFrame
        A dataframe with ids of samples as index and glycemia risks as columns.
    """
    df = glucose_data_verification(df)
    intervals = [54, 70, 180, 250]
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()
    glucose_signals = df_copy.groupby(level=0, sort=False)

    if 0 not in intervals:
        ranges = [0] + intervals
    else:
        ranges = intervals
    ranges = glucose_signals[column_name_glucose].max().apply(
        lambda x: ranges + [x] if x > ranges[-1] else ranges + [ranges[-1] + 1.])
    df_copy['ranges'] = glucose_signals.apply(
        lambda signal: pd.cut(signal[column_name_glucose], bins=ranges.loc[signal.name])).values

    time_ranges = pd.Series(df_copy['ranges'])
    map_ranges = list(time_ranges.apply(
        lambda interval:
        'vlow' if interval.right <= 54 else
        'low' if 54 < interval.right <= 70 else
        'normal' if 70 < interval.right <= 180 else
        'high' if 180 < interval.right <= 250 else
        'vhigh')
    )
    df_copy['ranges'] = map_ranges

    df_copy['time_diff'] = glucose_signals.apply(
            lambda signal: signal['time'].diff().dt.total_seconds()).values

    total_time = glucose_signals['time_diff'].sum()
    df_copy = df_copy[df_copy['ranges'] != 'normal']
    time_sum = df_copy.groupby([df_copy.index, 'ranges'], observed=False, sort=False)['time_diff'].sum()
    percentage_time = time_sum.div(total_time, level=0) * 100

    gr_df = percentage_time.unstack(level='ranges').fillna(0)
    for stat in ['vlow', 'vhigh', 'low', 'high']:
        if stat not in gr_df.columns:
            gr_df[stat] = [0] * len(gr_df)

    gr_df['gri'] = 3 * gr_df['vlow'] + 2.4 * gr_df['low'] + 1.6 * gr_df['high'] + 0.8 * gr_df['vhigh']
    gr_df['gri'] = gr_df['gri'].apply(lambda x: 100 if x > 100 else x)

    return gr_df


def grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Glycemic Risk Assessment Diabetes Equation (GRADE):
        - Glycemic Risk Assessment in Diabetes Equation (grade).
        - Percentage Glycemic Risk Assessment in Diabetes Equation in hypoglycemia (grade_hypo).
        - Percentage Glycemic Risk Assessment in Diabetes Equation in euglycemia (grade_eu).
        - Percentage Glycemic Risk Assessment in Diabetes Equation in hyperglycemia (grade_hyper).


    Parameters
    ----------
    df : pd.DataFrame MultiIndex
        A pd.DataFrame where the index is the unique identifier of the signals which must be an integer or a string, and
        it has two columns: the first column must contain the timestamps in datetime format of the samples and the
        second column must contain the glucose levels in mg/dL of the samples.

    Returns
    -------
    grade_df : pandas.DataFrame
        A dataframe with ids of samples as index and grade stats as columns.
    """
    df = glucose_data_verification(df)
    column_name_timestamps, column_name_glucose = df.columns

    df_copy = df.copy()

    grade_values = df_copy[column_name_glucose].apply(
        lambda x: np.minimum(425 * np.square(np.log10(np.log10(x/18)) + 0.16), 50))
    grade_hypo = grade_values[df_copy[column_name_glucose]/18 < 3.9]
    grade_hyper = grade_values[df_copy[column_name_glucose]/18 > 7.8]
    grade_eu = grade_values[(3.9 <= (df_copy[column_name_glucose] / 18)) & ((df_copy[column_name_glucose] / 18) <= 7.8)]

    patients_grades = grade_values.groupby(level=0, sort=False)
    patients_grades_hypo = grade_hypo.groupby(level=0, sort=False)
    patients_grades_hyper = grade_hyper.groupby(level=0, sort=False)
    patients_grades_eu = grade_eu.groupby(level=0, sort=False)

    grade_sum = patients_grades.sum()
    grade_mean = patients_grades.mean()

    grade_df = pd.DataFrame()
    grade_df['grade'] = grade_mean
    grade_df['grade_hypo'] = patients_grades_hypo.sum() / grade_sum * 100
    grade_df['grade_hyper'] = patients_grades_hyper.sum() / grade_sum * 100
    grade_df['grade_eu'] = patients_grades_eu.sum() / grade_sum * 100

    return grade_df.fillna(0)
