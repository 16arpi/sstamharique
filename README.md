# # አማራ

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An STT school project for Amharic

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Exploratory marimo notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         stt_amh and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── stt_amh   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes stt_amh a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    └─── modeling                
        ├── __init__.py 
        ├── predict.py          <- Code to run model inference with trained models          
        └── train.py            <- Code to train models
```

--------

## Installation

Tested with Python 3.13

## Environment setup

Install the required dependencies (using [uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
make create_environment
source .venv/bin/activate
make requirements
```

Pull the ready-to-go dataset:
```bash
make download_data
```

----

## Command-Line Usage

Training an adapter:
```bash
python stt_amh/modeling/train.py
```
