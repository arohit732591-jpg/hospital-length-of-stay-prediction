# 🏥 Hospital Length-of-Stay Range Prediction

## 1. Project Overview

This project uses machine learning classification models to predict a patient's hospital length-of-stay category based on hospital, admission, and patient information.

The target variable represents duration ranges rather than an exact number of days.

## 2. Business Objective

Hospitals can use stay-range estimates to support early bed availability and staffing planning.

This project compares classification models and identifies features associated with their predictions. The output is intended as a planning aid and should be reviewed alongside clinical judgment and operational information.

## 3. Dataset Description

The dataset contains **318,438 hospital admission records and 18 columns**.

**Target variable:** `Stay`

The target contains 11 length-of-stay categories:

- 0-10
- 11-20
- 21-30
- 31-40
- 41-50
- 51-60
- 61-70
- 71-80
- 81-90
- 91-100
- More than 100 Days

The source label `Nov-20` is standardized to `11-20` in the notebook.

### Main Features

**Hospital and Location**
- Hospital_code
- Hospital_type_code
- City_Code_Hospital
- Hospital_region_code

**Admission and Care**
- Department
- Ward_Type
- Ward_Facility_Code
- Bed_Grade
- Type_of_Admission
- Severity_of_Illness

**Patient and Facilities**
- Age
- Visitors_with_Patient
- Available_Extra_Rooms_in_Hospital

**Financial**
- Admission_Deposit

The identifier columns `case_id` and `patientid` are excluded from model training.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 5. Project Workflow

1. Data loading and initial exploration
2. Missing-value and duplicate inspection
3. Data cleaning and label standardization
4. Exploratory Data Analysis (EDA)
5. Feature and target separation
6. Stratified train-test split
7. Data preprocessing and one-hot encoding
8. Decision Tree model development
9. Random Forest model development
10. Model evaluation and comparison
11. Feature importance analysis
12. Business interpretation and conclusion

## 6. Data Preprocessing

The notebook applies the following preprocessing techniques:

- Standardizes the `Nov-20` label to `11-20` in the `Stay` and `Age` columns.
- Uses median imputation for missing numerical values.
- Uses most-frequent-value imputation for missing categorical values.
- Applies one-hot encoding to categorical features.
- Excludes `case_id` and `patientid` from predictive features.
- Uses a stratified 80:20 train-test split.

Preprocessing is implemented through Scikit-learn pipelines and fitted using training data.

## 7. Machine Learning Models

Two tree-based classification models are compared:

**Decision Tree Classifier**
- Uses a single decision tree.
- Configured with balanced class weights.
- Uses random state 42.

**Random Forest Classifier**
- Uses an ensemble of 100 decision trees.
- Configured with balanced class weights.
- Uses parallel processing with `n_jobs=-1`.
- Uses random state 42.

## 8. Model Evaluation

The models are evaluated using:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Classification report
- Confusion matrix

### Model Comparison

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Decision Tree | 0.294 | 0.216 | 0.296 |
| Random Forest | 0.375 | 0.263 | 0.359 |

**Selected candidate: Random Forest**

Random Forest achieved better performance across all three reported metrics on the held-out test set.

Its accuracy was 37.5%, with a macro F1-score of 0.263.

The results indicate that predicting all stay categories remains challenging, particularly for less frequent categories.

## 9. Feature Importance

Random Forest feature importance was used to examine the features contributing most to the model's predictions.

The top features included:

- Admission_Deposit
- Visitors_with_Patient
- Available_Extra_Rooms_in_Hospital
- City_Code_Patient
- Age categories
- Department
- Severity_of_Illness
- Bed_Grade

Feature importance reflects patterns learned by the model and does not establish causation.

## 10. Business Applications

The model may support:

- Early bed availability planning
- Hospital staffing preparation
- Expected resource-demand planning
- Analysis of admission and patient patterns
- Identification of features associated with hospital stay ranges

Predictions should be used as planning estimates, not as clinical decisions.

## 11. Challenges and Limitations

- The target variable contains duration ranges rather than exact stay durations.
- The dataset has uneven class distributions.
- Missing values require separate numerical and categorical imputation.
- Categorical hospital and patient attributes require encoding.
- The model's overall accuracy and macro F1-score indicate substantial room for improvement.
- Healthcare outcomes may depend on clinical factors not present in the dataset.
- The model requires validation on newer admissions and across different hospitals and patient groups before operational use.

## 12. Project Structure

```text
hospital-length-of-stay-prediction/
│
├── hospital-length-of-stay-prediction.ipynb
├── HealthCareAnalytics.csv
├── README.md
└── requirements.txt
```

## 13. How to Run the Project

1. Clone or download this repository.
2. Install Python.
3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Keep the notebook and dataset in the same folder.
5. Open the notebook in Jupyter Notebook or VS Code.
6. Run the notebook cells in order.

## 14. Conclusion

This project demonstrates a machine learning classification workflow for predicting hospital length-of-stay ranges.

Random Forest performed better than Decision Tree on the held-out test set, achieving 37.5% accuracy and a macro F1-score of 0.263.

Although Random Forest is selected as the candidate model, its performance requires improvement and further validation. The predictions should support operational planning only and must not replace clinical judgment.