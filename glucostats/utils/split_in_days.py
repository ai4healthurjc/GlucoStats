import pandas as pd
from glucostats.utils.format_verification import glucose_data_verification


def split_signals_by_day(df):
    df = glucose_data_verification(df)
    column_name_timestamps, column_name_glucose = df.columns

    df['date'] = df[column_name_timestamps].dt.date
    df['id'] = df.index.astype(str) + '_' + df['date'].astype(str)

    df = df.set_index('id').drop(['date'], axis=1)

    return df
