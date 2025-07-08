import pytest
import numpy as np
from glucostats.stats.glucose_variability import glucose_variability, signal_excursions


class TestGlucoseVariability:

    def test_output_structure(self, sample_glucose_data):
        result = glucose_variability(sample_glucose_data)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"mag", "gvp", "dt", "cv"}
        assert len(result) == len(sample_glucose_data.index.unique())

    def test_mag_calculation(self, sample_glucose_data):
        result = glucose_variability(sample_glucose_data)

        # Manual verification for id1
        group = sample_glucose_data.loc["id1"]
        diffs = np.abs(group["glucose"].diff().dropna())
        expected_mag = diffs.sum() / (group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]).total_seconds() * 3600

        assert np.isclose(result.loc["id1", "mag"], expected_mag, rtol=0.01)

    def test_cv_calculation(self, sample_glucose_data):
        result = glucose_variability(sample_glucose_data)

        # Compare with manual calculation
        group = sample_glucose_data.loc["id1"]
        std = group["glucose"].std()
        mean = group["glucose"].mean()
        expected_cv = (std / mean) * 100

        assert np.isclose(result.loc["id1", "cv"], expected_cv)

    def test_empty_input(self):
        empty_df = pd.DataFrame(columns=["timestamp", "glucose"])
        empty_df.index.name = "patient_id"

        result = glucose_variability(empty_df)
        assert result.empty

    def test_nan_handling(self):
        # Data with missing values
        timestamps = pd.date_range(start="2023-01-01", periods=5, freq="5min")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "glucose": [100, np.nan, 110, 105, np.nan]
        }).set_index(pd.Index(["nan_test"] * 5, name="patient_id"))

        result = glucose_variability(df)
        assert not np.isnan(result.loc["nan_test", "cv"])

    def test_single_point_input(self):
        # Edge case: only one data point
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2023-01-01")],
            "glucose": [100]
        }).set_index(pd.Index(["single"] * 1, name="patient_id"))

        result = glucose_variability(df)
        assert result.loc["single", "dt"] == 0
        assert np.isnan(result.loc["single", "cv"])  # Undefined for single point


class TestSignalExcursions:
    def test_mage_calculation(self, sample_glucose_data):
        result = signal_excursions(sample_glucose_data)

        # Basic output validation
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"mage", "ef"}
        assert not result["mage"].isnull().all()

    def test_ef_with_known_pattern(self):
        # Create data with exactly 3 peaks
        timestamps = pd.date_range(start="2023-01-01", periods=10, freq="5min")
        glucose = [100, 120, 110, 130, 115, 125, 110, 140, 115, 120]
        df = pd.DataFrame({
            "timestamp": timestamps,
            "glucose": glucose
        }).set_index(pd.Index(["test_id"] * 10, name="patient_id"))

        result = signal_excursions(df)
        assert result.loc["test_id", "ef"] == 3  # Expect 3 significant excursions

    def test_edge_case_flat_signal(self):
        # Zero-variation signal
        timestamps = pd.date_range(start="2023-01-01", periods=5, freq="5min")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "glucose": [100] * 5
        }).set_index(pd.Index(["flat"] * 5, name="patient_id"))

        result = signal_excursions(df)
        assert np.isnan(result.loc["flat", "mage"])  # No excursions
        assert result.loc["flat", "ef"] == 0