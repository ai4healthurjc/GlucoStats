import pandas as pd
from datetime import datetime, timedelta
from glucostats.stats.observations_stats import (
    observations_in_ranges,
    percentage_observations_in_ranges
)


def sample_data():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    ids = []
    timestamps = []
    glucose = []

    # id1: in range
    for i in range(10):
        ids.append("id1")
        timestamps.append(base_time + timedelta(minutes=i * 5))
        glucose.append(100)

    # id2: below range
    for i in range(5):
        ids.append("id2")
        timestamps.append(base_time + timedelta(minutes=i * 5))
        glucose.append(60)

    # id3: above range
    for i in range(6):
        ids.append("id3")
        timestamps.append(base_time + timedelta(minutes=i * 5))
        glucose.append(200)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "glucose": glucose
    }, index=ids)

    return df


def test_observations_in_ranges():
    df = sample_data()
    result = observations_in_ranges(df, in_range_interval=[70, 180])

    # id1: 10 in range
    assert result.loc["id1", "n_ir"] == 10
    assert result.loc["id1", "n_ar"] == 0
    assert result.loc["id1", "n_br"] == 0
    assert result.loc["id1", "n_or"] == 0

    # id2: 5 below range
    assert result.loc["id2", "n_ir"] == 0
    assert result.loc["id2", "n_ar"] == 0
    assert result.loc["id2", "n_br"] == 5
    assert result.loc["id2", "n_or"] == 5

    # id3: 6 above range
    assert result.loc["id3", "n_ir"] == 0
    assert result.loc["id3", "n_ar"] == 6
    assert result.loc["id3", "n_br"] == 0
    assert result.loc["id3", "n_or"] == 6


def test_percentage_observations_in_ranges():
    df = sample_data()
    result = percentage_observations_in_ranges(df, in_range_interval=[70, 180])

    # id1: 100% in range
    assert result.loc["id1", "pn_ir"] == 100
    assert result.loc["id1", "pn_or"] == 0

    # id2: 100% below
    assert result.loc["id2", "pn_br"] == 100
    assert result.loc["id2", "pn_or"] == 100

    # id3: 100% above
    assert result.loc["id3", "pn_ar"] == 100
    assert result.loc["id3", "pn_or"] == 100
