import pandas as pd
import importlib.resources as pkg_resources


def load_glucodata():
    with pkg_resources.open_text("glucostats.datasets.data", "glucodata.csv") as f:
        return pd.read_csv(f)

