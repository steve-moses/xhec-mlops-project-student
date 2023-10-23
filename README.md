# xhec-mlops-project-student

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
  - [Environment Setup](#environment-setup)
  - [Dependencies](#dependencies)
- [Data](#data)
- [Notebooks](#notebooks)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Modelling](#modelling)
- [Source Code](#source-code)
  - [Modelling](#modelling-1)
    - [Preprocessing](#preprocessing)
    - [Training](#training)
    - [Predicting](#predicting)
    - [Utilities](#utilities)
  - [Web Service](#web-service)
- [Continuous Integration](#continuous-integration)
- [Assets](#assets)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project is a part of the MLOps course at xHEC. The main goal of the project is to apply machine learning operations (MLOps) principles to a student dataset, ensuring that the model is reproducible, scalable, and maintainable.

## Setup

### Environment Setup
To set up the environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/steve-moses/xhec-mlops-project-student
```

# Navigate to the project directory
```bash
cd xhec-mlops-project-student
```
# Create a conda environment using the environment.yml file
```bash
conda env create -f environment.yml
```
# Activate the conda environment
```bash
conda activate xhec-mlops
```
# Install the dependencies
```bash
pip install -r requirements.txt
```
# Run the app 
```bash
python main.py
```
## Data
The dataset used in this project is the `abalone.csv` file located in the `data` directory. This dataset contains information about abalones, which are a type of marine mollusk. The dataset is used to predict the age of abalones based on various physical measurements.

## Notebooks

### Exploratory Data Analysis (EDA)
The `eda.ipynb` notebook located in the `notebooks` directory contains exploratory data analysis of the abalone dataset. This notebook provides insights into the distribution of data, relationships between different variables, and other important aspects that can help in building a machine learning model.

### Modelling
The `modelling.ipynb` notebook located in the `notebooks` directory contains the machine learning model built for predicting the age of abalones. This notebook includes data preprocessing, model training, and evaluation steps.

## Source Code

### Modelling

#### Preprocessing
The `preprocessing.py` file located in the `src/modelling` directory contains functions for preprocessing the abalone dataset.

#### Training
The `training.py` file located in the `src/modelling` directory contains functions for training the machine learning model.

#### Predicting
The `predicting.py` file located in the `src/modelling` directory contains functions for making predictions using the trained machine learning model.

#### Utilities
The `utils.py` file located in the `src/modelling` directory contains utility functions used in the modelling process.

### Web Service
The `web_service` directory located in the `src` directory contains code for a web service that can be used to make predictions using the trained machine learning model.

## Continuous Integration
The project uses GitHub Actions for continuous integration. The configuration file for continuous integration is located in the `.github/workflows/ci.yaml` file.

## Assets
The `assets` directory contains images used in the project, such as `PR_right.png` and `PR_wrong.png`.

## Contributing
If you would like to contribute to this project, please follow the guidelines outlined in the `CONTRIBUTING.md` file (to be created).

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
