# AI4I 2020 Predictive Maintenance — Machine Failure Prediction

## Overview

This project develops an end-to-end machine-learning workflow for predicting machine failures using the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

The project covers the complete supervised-learning workflow:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Train/test splitting
- Feature preprocessing using `scikit-learn` pipelines
- Training three classification models
- Stratified 5-fold cross-validation
- Model comparison using multiple evaluation metrics
- Final evaluation on a held-out test set
- Feature-importance analysis
- Discussion of limitations and future improvements

The main objective is to identify machine failures from operational measurements while paying particular attention to the **Failure** class, since missing a real machine failure can be more important than maximising overall accuracy.

---

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**Source:** UCI Machine Learning Repository

The dataset contains information about machine operating conditions and whether a machine failure occurred.

### Main features used

| Feature | Description |
|---|---|
| `Type` | Product/machine type |
| `Air temperature [K]` | Air temperature in Kelvin |
| `Process temperature [K]` | Process temperature in Kelvin |
| `Rotational speed [rpm]` | Rotational speed in revolutions per minute |
| `Torque [Nm]` | Torque in Newton metres |
| `Tool wear [min]` | Tool wear time in minutes |

### Target variable

`Machine failure`

- `0` — No machine failure
- `1` — Machine failure

The identifier columns (`UDI` and `Product ID`) were excluded from modelling.

The failure-mode indicator variables (`TWF`, `HDF`, `PWF`, `OSF`, and `RNF`) were also excluded from the main predictive model because they are closely related to failure events and may not represent information that would be available early enough in a real predictive-maintenance setting.

---

## Project Workflow

### 1. Exploratory Data Analysis

The dataset was first inspected to understand:

- Dataset dimensions
- Data types
- Missing values
- Duplicate observations
- Descriptive statistics
- Target-class distribution
- Feature distributions
- Relationships between variables

The target distribution showed that machine failures are relatively uncommon compared with normal operation. Therefore, the problem is an **imbalanced binary classification problem**.

---

### 2. Data Preprocessing

The preprocessing workflow was implemented using `scikit-learn` pipelines.

Numerical features were:

- Imputed using the median where necessary
- Standardised using `StandardScaler`

The categorical `Type` feature was:

- Imputed using the most frequent value where necessary
- Encoded using `OneHotEncoder`

A `ColumnTransformer` was used to apply the appropriate preprocessing to each feature type.

Using a pipeline ensures that preprocessing is performed correctly inside each cross-validation fold and helps prevent data leakage.

---

### 3. Models

Three classification algorithms were compared:

#### Logistic Regression

Used as an interpretable baseline model.

#### Random Forest

An ensemble of decision trees capable of modelling nonlinear relationships.

Class weighting was used to give additional importance to the minority failure class.

#### Gradient Boosting

A boosting-based ensemble method that sequentially improves weak learners to produce a stronger predictive model.

---

## Model Evaluation

The models were evaluated using **stratified 5-fold cross-validation**.

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Because the dataset is imbalanced, **recall, F1-score and ROC-AUC were given particular attention** rather than relying on accuracy alone.

### Why recall matters

In predictive maintenance, a false negative means that a machine that actually fails is predicted as healthy.

Therefore, improving the ability to detect the `Failure` class is particularly important.

---

## Results

Based on the cross-validation results, Gradient Boosting provided the strongest overall performance.

Approximate cross-validation results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.97 | 0.76 | 0.20 | 0.31 | 0.89 |
| Random Forest | 0.98 | 0.89 | 0.37 | 0.53 | 0.97 |
| **Gradient Boosting** | **0.98** | **0.86** | **0.60** | **0.70** | **0.98** |

Gradient Boosting achieved the highest recall, F1-score and ROC-AUC in cross-validation, making it the strongest overall model among the three evaluated approaches.

### Held-out test set

The classification reports showed the following performance for the **Failure** class:

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.64 | 0.10 | 0.18 |
| Random Forest | 0.91 | 0.46 | 0.61 |
| **Gradient Boosting** | **0.90** | **0.63** | **0.74** |

Gradient Boosting detected approximately **63% of the actual machine failures** in the held-out test set.

This was substantially better than Logistic Regression (10%) and Random Forest (46%).

---

## Feature Importance

Feature importance was analysed to identify which operational variables contributed most strongly to the model's predictions.

The main candidate predictive variables were:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type

The exact ranking of feature importance should be interpreted from the feature-importance output generated in the notebook.

---

## Conclusion

The project demonstrates an end-to-end machine-learning workflow for predictive maintenance using `scikit-learn`.

Among the three evaluated models, **Gradient Boosting provided the strongest overall performance**, achieving approximately **0.98 ROC-AUC** and **0.70 F1-score** in cross-validation. On the held-out test set, it achieved a **0.63 recall** and **0.74 F1-score** for the Failure class.

The results demonstrate that machine-learning models can identify patterns associated with machine failure from operational measurements. However, the model still misses some actual failures, meaning that further optimisation would be useful for a real predictive-maintenance application.

---

## Limitations and Future Work

The results should be interpreted with caution because:

- The AI4I 2020 dataset is relatively small.
- The dataset is simulated rather than collected from a real industrial production environment.
- Class imbalance makes accuracy an insufficient measure of model quality.
- The current models were not extensively hyperparameter-tuned.
- A default classification threshold of 0.5 was used.

Possible future improvements include:

1. Hyperparameter optimisation using `GridSearchCV` or `RandomizedSearchCV`.
2. Classification-threshold tuning to prioritise failure recall.
3. Additional feature engineering.
4. Calibration of predicted probabilities.
5. Cost-sensitive learning based on the real-world cost of missed failures.
6. Validation on real industrial sensor data.
7. Comparison with additional models such as XGBoost or other gradient-boosting approaches.

---

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

---

## Project Structure

```text
.
├── README.md
├── ai4i2020.csv
└── predictive_maintenance.ipynb
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
predictive_maintenance.ipynb
```

Make sure `ai4i2020.csv` is located in the same directory as the notebook.

---

## Author

**Tünzalə**

Machine Learning / AI for Engineering
