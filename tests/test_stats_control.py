import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the functions to test
from glucostats.utils.glucose_analysis import g_control, a1c_estimation, qgc_index


class TestGlycemicControlFunctions(unittest.TestCase):
    def setUp(self):
        """Create test data that will be used across multiple test cases"""
        # Create sample data with 3 different signals (patients/days)
        timestamps1 = [datetime(2023, 1, 1) + timedelta(minutes=5 * i) for i in range(10)]
        glucose1 = [80, 85, 90, 100, 110, 120, 130, 140, 150, 160]  # All in range

        timestamps2 = [datetime(2023, 1, 2) + timedelta(minutes=5 * i) for i in range(10)]
        glucose2 = [60, 65, 70, 75, 180, 185, 190, 195, 200, 205]  # Mix of below and above

        timestamps3 = [datetime(2023, 1, 3) + timedelta(minutes=5 * i) for i in range(10)]
        glucose3 = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290]  # All above range

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

    def test_g_control_default_params(self):
        """Test g_control with default parameters"""
        result = g_control(self.test_df)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertListEqual(sorted(result.columns),
                             sorted(['hypo_index', 'hyper_index', 'igc']))

        # Check values for first signal (all in range)
        self.assertAlmostEqual(result.loc[1, 'hypo_index'], 0.0)
        self.assertAlmostEqual(result.loc[1, 'hyper_index'], 0.0)
        self.assertAlmostEqual(result.loc[1, 'igc'], 0.0)

        # Check second signal (4 below, 5 above range)
        # Calculate expected values manually
        lltr, ultr = 70, 180
        b, a = 2.0, 1.1
        c, d = 30, 30

        # Hypo index calculation
        below_values = [60, 65, 70, 75]
        hypo_sum = sum((lltr - x) ** b for x in below_values if x < lltr)  # 70 is not below range
        expected_hypo = hypo_sum / (d * 10)
        self.assertAlmostEqual(result.loc[2, 'hypo_index'], expected_hypo)

        # Hyper index calculation
        above_values = [185, 190, 195, 200, 205]
        hyper_sum = sum((x - ultr) ** a for x in above_values)
        expected_hyper = hyper_sum / (c * 10)
        self.assertAlmostEqual(result.loc[2, 'hyper_index'], expected_hyper)

        # IGC is sum of both
        self.assertAlmostEqual(result.loc[2, 'igc'], expected_hypo + expected_hyper)

        # Check third signal (all above range)
        above_values = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290]
        hyper_sum = sum((x - ultr) ** a for x in above_values)
        expected_hyper = hyper_sum / (c * 10)
        self.assertAlmostEqual(result.loc[3, 'hyper_index'], expected_hyper)
        self.assertAlmostEqual(result.loc[3, 'hypo_index'], 0.0)
        self.assertAlmostEqual(result.loc[3, 'igc'], expected_hyper)

    def test_g_control_custom_params(self):
        """Test g_control with custom parameters"""
        a, b = 1.5, 1.8
        c, d = 25, 35
        result = g_control(self.test_df, a=a, b=b, c=c, d=d)

        # Check second signal values with custom parameters
        lltr, ultr = 70, 180

        # Hypo index calculation
        below_values = [60, 65, 70, 75]
        hypo_sum = sum((lltr - x) ** b for x in below_values if x < lltr)
        expected_hypo = hypo_sum / (d * 10)
        self.assertAlmostEqual(result.loc[2, 'hypo_index'], expected_hypo)

        # Hyper index calculation
        above_values = [185, 190, 195, 200, 205]
        hyper_sum = sum((x - ultr) ** a for x in above_values)
        expected_hyper = hyper_sum / (c * 10)
        self.assertAlmostEqual(result.loc[2, 'hyper_index'], expected_hyper)

    def test_g_control_invalid_params(self):
        """Test g_control with invalid parameters"""
        with self.assertRaises(ValueError):
            g_control(self.test_df, a='invalid')
        with self.assertRaises(ValueError):
            g_control(self.test_df, b='invalid')
        with self.assertRaises(ValueError):
            g_control(self.test_df, c='invalid')
        with self.assertRaises(ValueError):
            g_control(self.test_df, d='invalid')

    def test_a1c_estimation(self):
        """Test a1c_estimation function"""
        result = a1c_estimation(self.test_df)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertListEqual(sorted(result.columns),
                             sorted(['gmi', 'eA1C']))

        # Check values for first signal
        mean_glucose = np.mean(range(80, 161, 10))
        expected_gmi = 3.31 + 0.02392 * mean_glucose
        expected_ea1c = (mean_glucose + 46.7) / 28.7
        self.assertAlmostEqual(result.loc[1, 'gmi'], expected_gmi, places=4)
        self.assertAlmostEqual(result.loc[1, 'eA1C'], expected_ea1c, places=4)

        # Check second signal
        mean_glucose = np.mean([60, 65, 70, 75, 180, 185, 190, 195, 200, 205])
        expected_gmi = 3.31 + 0.02392 * mean_glucose
        expected_ea1c = (mean_glucose + 46.7) / 28.7
        self.assertAlmostEqual(result.loc[2, 'gmi'], expected_gmi, places=4)
        self.assertAlmostEqual(result.loc[2, 'eA1C'], expected_ea1c, places=4)

    def test_qgc_index_default(self):
        """Test qgc_index with default ideal_bg"""
        result = qgc_index(self.test_df)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertListEqual(sorted(result.columns),
                             sorted(['m_value', 'j_index']))

        # Check values for first signal
        signal_values = np.array(range(80, 161, 10))
        ideal_bg = 120
        m_values = np.abs(10 * np.log10(signal_values / ideal_bg)) ** 3
        expected_m = np.mean(m_values) + ((signal_values.max() - signal_values.min()) / 20)

        mean = signal_values.mean()
        std = signal_values.std()
        expected_j = 0.001 * ((mean + std) ** 2)

        self.assertAlmostEqual(result.loc[1, 'm_value'], expected_m, places=4)
        self.assertAlmostEqual(result.loc[1, 'j_index'], expected_j, places=4)

    def test_qgc_index_custom_ideal(self):
        """Test qgc_index with custom ideal_bg"""
        ideal_bg = 100
        result = qgc_index(self.test_df, ideal_bg=ideal_bg)

        # Check first signal values
        signal_values = np.array(range(80, 161, 10))
        m_values = np.abs(10 * np.log10(signal_values / ideal_bg)) ** 3
        expected_m = np.mean(m_values) + ((signal_values.max() - signal_values.min()) / 20)

        self.assertAlmostEqual(result.loc[1, 'm_value'], expected_m, places=4)

    def test_qgc_index_invalid_ideal(self):
        """Test qgc_index with invalid ideal_bg"""
        with self.assertRaises(ValueError):
            qgc_index(self.test_df, ideal_bg='invalid')


if __name__ == '__main__':
    unittest.main()