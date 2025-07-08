import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_glucose_data():
    """
    Generate synthetic CGM data for testing
    """

    base_time = datetime(2023, 1, 1, 8, 0, 0)
    data = {
        "id1": [
            (base_time + timedelta(minutes=5 * i), 100 + 10 * np.sin(i))
            for i in range(24)  # 2 hours of 5-min data
        ],
        "id2": [
            (base_time + timedelta(minutes=10 * i), 90 + 5 * (-1) ** i)
            for i in range(12)  # 2 hours of 10-min data
        ]
    }

    dfs = []
    for pid, values in data.items():
        timestamps, glucose = zip(*values)
        dfs.append(pd.DataFrame({
            "timestamp": timestamps,
            "glucose": glucose
        }).assign(patient_id=pid))

    combined = pd.concat(dfs).set_index("patient_id")

    return combined