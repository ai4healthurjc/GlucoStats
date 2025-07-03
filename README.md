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




