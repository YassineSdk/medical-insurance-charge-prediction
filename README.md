# Medical Insurance Charge Prediction

An exploratory data analysis and regression project for predicting medical insurance charges. The project uses demographic, lifestyle, regional, occupational, and coverage data to identify charge drivers and compare Random Forest and XGBoost models.

## Dataset

The source file is `insurance_data.csv`, with 561,672 rows and 12 analytical columns after removing the index column.

| Column | Description |
|---|---|
| `age` | Insured person's age |
| `gender` | Gender |
| `bmi` | Body Mass Index |
| `children` | Number of covered dependents |
| `smoker` | Smoking status |
| `region` | Residential region |
| `medical_history` | Personal medical history |
| `family_medical_history` | Family medical history |
| `exercise_frequency` | Exercise frequency |
| `occupation` | Occupation category |
| `coverage_level` | Insurance coverage level |
| `charges` | Insurance charge target |

The notebook also contains the original `Unnamed: 0` index column, which is removed during preprocessing.

## Analysis

The notebook:

1. Profiles the dataset and categorical/numerical distributions.
2. Groups categorical variables to compare average and variation in charges.
3. Removes charge outliers with a Z-score threshold of 3.
4. Examines relationships between age, BMI, dependents, and charges.
5. Encodes categorical variables and standardizes numerical features.
6. Trains and compares Random Forest and XGBoost regressors.

The analysis identifies smoking status, medical history, coverage level, age, BMI, and the number of dependents as important influences on insurance charges.

## Data Preparation

- Numerical features `age` and `bmi` are standardized with `StandardScaler`.
- The target `charges` is standardized before model training.
- Binary categorical features are encoded with `OneHotEncoder`.
- `medical_history`, `exercise_frequency`, `occupation`, and `coverage_level` use ordered encoding.
- The data is split 70/30 for training and testing with `random_state=42`.

## Models

| Model | MAE | MSE | R-squared |
|---|---:|---:|---:|
| Random Forest Regressor | 0.3792 | 0.1939 | 0.8060 |
| XGBoost Regressor | 0.3668 | 0.1702 | 0.8297 |

The XGBoost model records the stronger result in the notebook. Metrics are calculated against the standardized target values. Five-fold cross-validation of the Random Forest model produces an average MAE of approximately 0.3798.

## Requirements

The notebook uses:

- pandas
- numpy
- matplotlib
- seaborn
- skimpy
- tqdm
- scipy
- scikit-learn
- xgboost

Jupyter Notebook or JupyterLab is required to open and execute `model.ipynb`.

## Getting Started

Clone the repository and create a Python environment:

```bash
git clone https://github.com/YassineSdk/medical-insurance-charge-prediction.git
cd medical-insurance-charge-prediction
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn skimpy tqdm scipy scikit-learn xgboost jupyter
jupyter notebook model.ipynb
```

Run the notebook cells in order. The notebook reads `insurance_data.csv` relative to the repository directory.

## Project Structure

```text
.
├── insurance_data.csv  # Source dataset
├── model.ipynb         # Exploratory analysis and regression workflow
└── README.md           # Project documentation
```
