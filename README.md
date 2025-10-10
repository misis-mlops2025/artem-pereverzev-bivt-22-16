# project_name

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

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
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         project_name and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── project_name   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes project_name a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------



## Checks

```bash
uv run ruff check
```


```bash
uv run pytest tests/
```

# Project
DVC Pipeline
This project uses DVC (Data Version Control) for reproducible machine learning workflows.

## Pipeline Stages
The DVC pipeline consists of 5 stages:

- generate - Creates synthetic data
- preprocess - Data preprocessing and normalization
- train - Trains machine learning models
- evaluate - Evaluates model performance
- plots - Creates visualizations from metrics

## Quick Start
```Bash
# Initialize DVC (first time only)
dvc init

# Run the complete pipeline
dvc repro

# Run specific stages
dvc repro train evaluate

# Show pipeline status
dvc status
```
## Configuration
All parameters are in params.yaml:

```Yaml
Apply
data:
  seed: 42
  n_samples: 1000

preprocess:
  scaler_type: "robust"

train:
  model_type: "random_forest"
  n_estimators: 100
```
## Manual Testing
```Bash
# Run pipeline without DVC
uv run python run_pipeline.py
```
The pipeline automatically handles data versioning, dependency tracking, and reproducible experiments.