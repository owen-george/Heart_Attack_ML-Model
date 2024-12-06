# Heart_Attack_ML-Model

## Overview

Heart diseases are leading factors of death in the United States, with millions of people affected annually. Early diagnosis and prevention can significantly reduce the risk of severe outcomes like heart attacks. The Behavioral Risk Factor Surveillance System (BRFSS) is a large-scale health survey conducted by the Centers for Disease Control and Prevention (CDC) that collects data on health behaviours, chronic health conditions, and preventive services. In this project BRFSS dataset from 2015 was used to explore and predict heart disease risk and investigate if a subset of survey response can serve as a useful tool for preventive health screening.

## Dataset

The dataset used for this project is a cleaned and consolidated version of the 2015 BRFSS dataset, sourced from Kaggle. It contains survey responses from over 250,000 individuals across the United States. The dataset includes a mix of health behaviors, chronic health conditions, demographic information, and medical history collected through a telephone survey.

## Requirements

Python, pandas, numpy, sklearn, matplotlib, seaborn

### Key Features

- **Health Metrics**: Blood pressure, cholesterol levels, BMI, and lifestyle factors
- **Chronic Conditions**: Diabetes, stroke history, walking difficulties
- **Healthcare Access**: Coverage, cost-related barriers to accessing healthcare in the past year
- **Demographic Data**: Age, sex, education level, household income

## Data Cleaning

Several features were removed during data cleaning to reduce multicollinearity and improve model performance:
- **PhysHlth** and **DiffWalk**: Removed due to high correlation with **GenHlth**
- **Education**: Removed due to high correlation with **Income**

### Feature and Target Separation

- **Features**: All columns except the target variable `HeartDiseaseorAttack`
- **Target**: `HeartDiseaseorAttack` column (binary: 0 = No, 1 = Yes)

## Data Preparation

### Train-Test Split

- The dataset was split into training and testing sets, with 80% used for training and 20% for testing (`test_size=0.2`), ensuring reproducibility with a fixed random state.

### Feature Normalization

- **Min-Max Scaling** was applied to numerical features to scale all features between 0 and 1. This ensures that all features contribute equally during distance calculations in the K-Nearest Neighbors (KNN) algorithm.

### Resampling

- **Class Imbalance**: The target variable `HeartDiseaseorAttack` showed significant imbalance (~90% instances labeled as "No").
- **KNN Bias**: Initial evaluation with KNN showed high accuracy (~90%), but this was biased due to the class imbalance.
- **Resampling Strategy**: The majority class (0) was undersampled to match the minority class (1), creating a balanced dataset for further machine learning applications.

## Model Training & Evaluation

### K-Nearest Neighbors (KNN) Classifier

The following steps were performed to train and evaluate the KNN classifier:

- **Baseline Model (k=3)**: The model was trained using unnormalized features as a baseline. Accuracy was reported on both the training and test sets.
- **Optimized Model (k=35)**: The model was trained using normalized features and evaluated with accuracy, Cohen's Kappa, and recall scores to assess the impact of normalization and hyperparameter tuning.

### Performance Metrics

- **Accuracy**: Measures the overall correctness of the model.
- **Cohen's Kappa**: A measure of agreement between predicted and actual values, accounting for chance agreement.
- **Recall**: Important for medical diagnoses, recall measures the ability to correctly identify positive cases (i.e., individuals with heart disease).

## Key Insights

- The **recall score** was prioritized due to the medical context, where false negatives (failing to identify heart disease) can have severe consequences. Thus, the recall score was used alongside accuracy and Cohen's Kappa for a more robust model evaluation.
- All ensemble methods achieved high recall (around 80%).
- **AdaBoost** had the highest Cohen's Kappa (0.55)
- **Random Forest** showed slightly stronger recall (0.80 vs. 0.79).
- **Bayesian-optimized AdaBoost and Random Forest** models had strong recall on training data but underperformed on test data, suggesting overfitting.

## Presentation

https://www.canva.com/design/DAGYVjYO_IA/qOpBjyddozmdtBv7GV4IgQ/edit

## Team

Eliska
Lorena
Camil
Owen
Filip
