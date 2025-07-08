import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the functions to test (assuming they're in a module named glucose_analysis)
from glucostats.utils.glucose_analysis import mean_in_ranges, distribution, complexity, auc


class TestGlucoseAnalysis(unittest.TestCase):
    def setUp(self):
        """Create test data that will be used across multiple test cases"""
        # Create sample data with 3 different signals (patients/days)
        timestamps1 = [datetime(2023, 1, 1) + timedelta(minutes=5 * i) for i in range(10)]
        glucose1 = [80, 85, 90, 100, 110, 120, 130, 140, 150, 160]

        timestamps2 = [datetime(2023, 1, 2) + timedelta(minutes=5 * i) for i in range(10)]
        glucose2 = [60, 65, 70, 75, 180, 185, 190, 195, 200, 205]

        timestamps3 = [datetime(2023, 1, 3) + timedelta(minutes=5 * i) for i in range(10)]
        glucose3 = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290]

        # Create MultiIndex DataFrame
        index = pd.MultiIndex.from_arrays([
            [1] * 10 + [2] * 10 + [3] * 10,  # IDs
            timestamps1 + timestamps2 + timestamps3  # Timestamps
        ], names=['id', 'time'])

        self.test_df = pd.DataFrame({
            'glucose': glucose1 + glucose2 + glucose3
        }, index=index)

        # Create a single signal DataFrame for some tests
        self.single_df = self.test_df.loc[1].copy()
        self.single_df.index = pd.MultiIndex.from_arrays([
            [1] * 10,
            timestamps1
        ], names=['id', 'time'])

    def test_mean_in_ranges_default_range(self):
        """Test mean_in_ranges with default range [70, 180]"""
        result = mean_in_ranges(self.test_df)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # 3 signals
        self.assertListEqual(sorted(result.columns),
                             sorted(['mean_ir', 'mean_ar', 'mean_br', 'mean_or']))

        # Check values for first signal (all in range)
        self.assertAlmostEqual(result.loc[1, 'mean_ir'], np.mean(range(80, 161, 10)))
        self.assertTrue(np.isnan(result.loc[1, 'mean_ar']))
        self.assertTrue(np.isnan(result.loc[1, 'mean_br']))
        self.assertTrue(np.isnan(result.loc[1, 'mean_or']))  # No out of range values

        # Check second signal (mix of below and above range)
        self.assertAlmostEqual(result.loc[2, 'mean_ir'], 180)  # Only one value in range
        self.assertAlmostEqual(result.loc[2, 'mean_ar'], np.mean(range(185, 206, 5)))
        self.assertAlmostEqual(result.loc[2, 'mean_br'], np.mean(range(60, 76, 5)))
        self.assertAlmostEqual(result.loc[2, 'mean_or'], np.mean([60, 65, 70, 75, 185, 190, 195, 200, 205]))

        # Check third signal (all above range)
        self.assertTrue(np.isnan(result.loc[3, 'mean_ir']))
        self.assertAlmostEqual(result.loc[3, 'mean_ar'], np.mean(range(200, 291, 10)))
        self.assertTrue(np.isnan(result.loc[3, 'mean_br']))
        self.assertAlmostEqual(result.loc[3, 'mean_or'], np.mean(range(200, 291, 10)))

    def test_mean_in_ranges_custom_range(self):
        """Test mean_in_ranges with custom range [80, 150]"""
        result = mean_in_ranges(self.test_df, [80, 150])

        # Check values for first signal
        self.assertAlmostEqual(result.loc[1, 'mean_ir'], np.mean([80, 85, 90, 100, 110, 120, 130, 140, 150]))
        self.assertAlmostEqual(result.loc[1, 'mean_ar'], 160)
        self.assertTrue(np.isnan(result.loc[1, 'mean_br']))
        self.assertAlmostEqual(result.loc[1, 'mean_or'], 160)  # Only one value out of range

        # Check second signal
        self.assertAlmostEqual(result.loc[2, 'mean_ir'], np.mean([180]))  # Only one value in range
        self.assertAlmostEqual(result.loc[2, 'mean_ar'], np.mean([185, 190, 195, 200, 205]))
        self.assertAlmostEqual(result.loc[2, 'mean_br'], np.mean([60, 65, 70, 75]))
        self.assertAlmostEqual(result.loc[2, 'mean_or'], np.mean([60, 65, 70, 75, 185, 190, 195, 200, 205]))

    def test_mean_in_ranges_single_signal(self):
        """Test mean_in_ranges with a single signal"""
        result = mean_in_ranges(self.single_df)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[1, 'mean_ir'], np.mean(range(80, 161, 10)))
        self.assertTrue(np.isnan(result.loc[1, 'mean_ar']))
        self.assertTrue(np.isnan(result.loc[1, 'mean_br']))
        self.assertTrue(np.isnan(result.loc[1, 'mean_or']))

    def test_distribution_default(self):
        """Test distribution function with default parameters"""
        result = distribution(self.test_df)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        expected_cols = ['max', 'min', 'max_diff', 'mean', 'std',
                         'quartile_0.25', 'quartile_0.5', 'quartile_0.75', 'iqr']
        self.assertListEqual(sorted(result.columns), sorted(expected_cols))

        # Check values for first signal
        self.assertEqual(result.loc[1, 'max'], 160)
        self.assertEqual(result.loc[1, 'min'], 80)
        self.assertEqual(result.loc[1, 'max_diff'], 80)
        self.assertAlmostEqual(result.loc[1, 'mean'], np.mean(range(80, 161, 10)))
        self.assertAlmostEqual(result.loc[1, 'std'], np.std(range(80, 161, 10), delta=0.1)

        # Check quartiles (values are equally spaced so quartiles should be exact)
        self.assertEqual(result.loc[1, 'quartile_0.25'], 95)
        self.assertEqual(result.loc[1, 'quartile_0.5'], 120)
        self.assertEqual(result.loc[1, 'quartile_0.75'], 145)
        self.assertEqual(result.loc[1, 'iqr'], 50)

    def test_distribution_custom_quantiles(self):
        """Test distribution function with custom quantiles"""
        result = distribution(self.test_df, qs=[0.1, 0.5, 0.9])

        # Check structure
        self.assertIn('quartile_0.1', result.columns)
        self.assertIn('quartile_0.5', result.columns)
        self.assertIn('quartile_0.9', result.columns)
        self.assertNotIn('quartile_0.25', result.columns)
        self.assertNotIn('quartile_0.75', result.columns)
        self.assertIn('iqr', result.columns)  # Should still calculate IQR

        # Check values for first signal
        self.assertEqual(result.loc[1, 'quartile_0.1'], 83)
        self.assertEqual(result.loc[1, 'quartile_0.5'], 120)
        self.assertEqual(result.loc[1, 'quartile_0.9'], 157)

    def test_distribution_ddof_0(self):
        """Test distribution function with ddof=0"""
        result = distribution(self.test_df, ddof=0)
        std_ddof1 = np.std(range(80, 161, 10))
        std_ddof0 = np.std(range(80, 161, 10), ddof=0)
        self.assertAlmostEqual(result.loc[1, 'std'], std_ddof0, delta=0.1)
        self.assertNotAlmostEqual(result.loc[1, 'std'], std_ddof1, delta=0.1)

    def test_distribution_invalid_inputs(self):
        """Test distribution function with invalid inputs"""
        with self.assertRaises(ValueError):
            distribution(self.test_df, ddof=2)

        with self.assertRaises(TypeError):
            distribution(self.test_df, qs="not a list")

        with self.assertRaises(ValueError):
            distribution(self.test_df, qs=[0.5, 1.5])

    def test_complexity_structure(self):
        """Test that complexity returns expected structure"""
        result = complexity(self.test_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertListEqual(sorted(result.columns), sorted(['dfa', 'entropy']))

        # We can't easily test the actual values since they're complex calculations
        # but we can verify they're numbers
        self.assertTrue(np.isscalar(result.loc[1, 'dfa']))
        self.assertTrue(np.isscalar(result.loc[1, 'entropy']))

    def test_complexity_single_signal(self):
        """Test complexity with a single signal"""
        result = complexity(self.single_df)

        self.assertEqual(len(result), 1)
        self.assertTrue(np.isscalar(result.loc[1, 'dfa']))
        self.assertTrue(np.isscalar(result.loc[1, 'entropy']))

    def test_auc_above_threshold(self):
        """Test AUC calculation above threshold"""
        result = auc(self.test_df, threshold=180, where='above')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn('auc', result.columns)

        # First signal has no values above 180
        self.assertEqual(result.loc[1, 'auc'], 0)

        # Second signal has values above 180 starting at index 4
        # We need to calculate expected AUC manually
        time_diff = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]) / 60  # hours
        glucose_diff = np.array([0, 0, 0, 0, 0, 5, 10, 15, 20, 25])
        expected_auc = np.trapz(glucose_diff, time_diff)
        self.assertAlmostEqual(result.loc[2, 'auc'], expected_auc, delta=0.1)

        # Third signal is all above 180
        time_diff = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]) / 60  # hours
        glucose_diff = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
        expected_auc = np.trapz(glucose_diff, time_diff)
        self.assertAlmostEqual(result.loc[3, 'auc'], expected_auc, delta=0.1)

    def test_auc_below_threshold(self):
        """Test AUC calculation below threshold"""
        result = auc(self.test_df, threshold=100, where='below')

        # First signal has values below 100 for first 4 points
        time_diff = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]) / 60  # hours
        glucose_diff = np.array([20, 15, 10, 0, 0, 0, 0, 0, 0, 0])
        expected_auc = np.trapz(glucose_diff, time_diff)
        self.assertAlmostEqual(result.loc[1, 'auc'], expected_auc, delta=0.1)

        # Second signal has values below 100 for first 4 points
        glucose_diff = np.array([40, 35, 30, 25, 0, 0, 0, 0, 0, 0])
        expected_auc = np.trapz(glucose_diff, time_diff)
        self.assertAlmostEqual(result.loc[2, 'auc'], expected_auc, delta=0.1)

        # Third signal has no values below 100
        self.assertEqual(result.loc[3, 'auc'], 0)

    def test_auc_invalid_inputs(self):
        """Test AUC with invalid inputs"""
        with self.assertRaises(TypeError):
            auc(self.test_df, threshold='invalid')

        with self.assertRaises(ValueError):
            auc(self.test_df, threshold=-10)

        with self.assertRaises(ValueError):
            auc(self.test_df, where='invalid')

