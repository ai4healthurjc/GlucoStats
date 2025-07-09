Glucostats
====

## Description
GlucoStats is a Python library that enables the extraction of a set of 59 statistics from Continuous Glucose Monitoring (CGM) devices
, each intended to characterize glucose time series for in-depth study and analysis. These statistics are systematically categorized into 6 categories and further divided 
into 16 subcategories, based on the nature of the statistics and the type of information they provide.The library also 
integrates with Matplotlib and Seaborn for generating high-quality visualizations, enabling users to conduct comprehensive 
visual analyses of glucose patterns with minimal effort. For users requiring high-performance processing, GlucoStats supports parallel 
computation through Python's module threads. As a result, It provides a comprehensive toolset for analyzing CGM data,
along with visualization tools designed to support either technical and non-technical users.

The full documentation including tutorials and feature description is available at: https://glucostats.readthedocs.io/en/latest/

## Installation and setup

You can use pip :
```console
pip install glucostats
```

If you want to download the source code, you can clone it from the GitHub repository.
```console
git clone git@github.com:ai4healthurjc/GlucoStats.git 
```

If you install the library from GitHub, make sure to install the required Python dependencies by running:
```console
pip install -r requirements.txt 
```

## Basic usage

```python
import pandas as pd
from glucostats.datasets import load_glucodata
from glucostats.extract_statistics import ExtractGlucoStats

# Load example of glucose data
df_data = load_glucodata()

# Change datetime format and set index for multiple signals
df_data.index = df_data['id']
df_data = df_data[['time', 'glucose']]
df_data["time"] = pd.to_datetime(df_data["time"], errors="coerce")

# Define list of statistics to compute
list_statistics = ['hypo_index', 'max_lbgi', 'mean']

# Define parameters for creating windows
windowing = True
windowing_method = 'number'
windowing_param = 4
windowing_start = 'tail'
windowing_overlap = False

# Instantiate ExtractGlucoStats class
stats_extraction = ExtractGlucoStats(
    list_statistics,
    windowing,
    windowing_method,
    windowing_param,
    windowing_start,
    windowing_overlap,
    batch_size=20,
    n_workers=4
)

# Configuration of intervals for hypoglycemia, euglycemia, hyperglycemia
in_range_interval = [70, 180]

# Class configuration and statistics extraction
stats_extraction.configuration(in_range_interval=in_range_interval)
df_cgm_stats = stats_extraction.transform(df_data)
```


## Run unit tests

To run all tests:
```shell
pytest -v
```

## Paper reference

If you use `GlucoStats` in your research papers, please refer to it using following reference:
```bibtex
@article{peiroglucostats,
  title={Glucostats: An Efficient Python Library for Glucose Time Series Feature Extraction and Visual Analysis},
  author={Peiro-Corbacho, Pablo and Lara-Abelenda, Francisco Jesus and Chushig-Muzo, David and W{\"a}gner, Ana M and Granja, Concei{\c{c}}{\~a}o and Soguero-Ruiz, Cristina},
  journal={Available at SSRN 5203999}
}
```

