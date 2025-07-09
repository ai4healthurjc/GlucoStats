import pytest
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_glucose_data():
    """
    Generate synthetic CGM data for testing with:
    - id1: Smooth sinusoidal pattern (lower variability)
    - id2: Alternating high-low pattern (higher variability, clear excursions)
    - id3: Flat line (no variability)
    """

    base_time = datetime(2023, 1, 1, 8, 0, 0)
    data = {
        "id1": [  # Very stable
            (base_time + timedelta(minutes=5*i), 100 + random.gauss(0, 2))
            for i in range(24)
        ],
        "id2": [  # Clear large excursions
            (base_time + timedelta(minutes=10*i),
             70 if i%2 else 130)  # Large changes of ±60
            for i in range(12)
        ],
        "id3": [  # Flat line
            (base_time + timedelta(minutes=15*i), 100)
            for i in range(8)
        ]
    }
    dfs = []
    for pid, values in data.items():
        timestamps, glucose = zip(*values)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "glucose": glucose
        })
        df.index = [pid] * len(df)
        dfs.append(df)

    combined = pd.concat(dfs)

    return combined
