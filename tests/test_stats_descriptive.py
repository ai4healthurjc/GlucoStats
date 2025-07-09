import numpy as np
import pytest


from glucostats.stats.descriptive_stats import mean_in_ranges, distribution, complexity, auc


@pytest.fixture
def sample_glucose_df():
    import pandas as pd
    from datetime import datetime, timedelta

    base_time = datetime(2023, 1, 1, 0, 0, 0)
    data = []

    for i in range(6):
        time = base_time + timedelta(minutes=10*i)
        data.append(("id1", time, 80 + i*5))
    for i in range(6):
        time = base_time + timedelta(minutes=10*i)
        data.append(("id2", time, 120 + i*10))
    for i in range(6):
        time = base_time + timedelta(minutes=10*i)
        data.append(("id3", time, 100))  # flat line

    df = pd.DataFrame(data, columns=["id", "timestamp", "glucose"])
    df.set_index(["id", "timestamp"], inplace=True)  # <-- MultiIndex con dos niveles
    return df



def test_mean_in_ranges_complex(sample_glucose_df):
    df = sample_glucose_df.copy()
    df.loc[("id1", slice(None)), "glucose"] = [60, 65, 75, 190, 195, 200]

    result = mean_in_ranges(df, in_range_interval=[70, 180])

    id1 = result.loc["id1"]
    assert np.isclose(id1["mean_ir"], 75)
    assert np.isclose(id1["mean_br"], np.mean([60, 65]))
    assert np.isclose(id1["mean_ar"], np.mean([190, 195, 200]))
    assert np.isclose(id1["mean_or"], np.mean([60, 65, 190, 195, 200]))


def test_distribution_stats_quantiles(sample_glucose_df):
    df = sample_glucose_df.copy()
    result = distribution(df, ddof=0, qs=[0.1, 0.5, 0.9])

    expected_cols = ['max', 'min', 'max_diff', 'mean', 'std',
                     'quartile_0.1', 'quartile_0.5', 'quartile_0.9', 'iqr']

    for col in expected_cols:
        assert col in result.columns
        assert result[col].notnull().all()

    expected_iqr = result["quartile_0.9"] - result["quartile_0.1"]
    assert np.allclose(result["iqr"], expected_iqr)


def test_complexity_metrics_robust(sample_glucose_df):
    df = sample_glucose_df.copy()
    df.loc[("id3", slice(None)), "glucose"] = [100] * 6  # flat line

    result = complexity(df)

    for metric in ["entropy", "dfa"]:
        assert metric in result.columns

    assert result["dfa"].between(0, 2).all()
    assert result["entropy"].ge(0).all()


def test_auc_threshold_logic(sample_glucose_df):
    df = sample_glucose_df.copy()
    df.loc[("id1", slice(None)), "glucose"] = [70, 75, 65, 80, 60, 90]  # mostly below 100
    df.loc[("id2", slice(None)), "glucose"] = [140, 150, 160, 170, 180, 190]  # above 100

    auc_above = auc(df, threshold=100, where="above")
    auc_below = auc(df, threshold=100, where="below")

    assert auc_below.loc["id1", "auc"] > auc_above.loc["id1", "auc"]
    assert auc_above.loc["id2", "auc"] > auc_below.loc["id2", "auc"]
