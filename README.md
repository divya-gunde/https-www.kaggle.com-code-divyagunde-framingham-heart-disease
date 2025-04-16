# 🫀 Framingham Heart Disease Prediction

This project utilizes the **Framingham Heart Study** dataset to predict the 10-year risk of coronary heart disease (CHD) using various machine learning models. The goal is to identify individuals at risk and facilitate early intervention.

## 📚 Overview

The Framingham Heart Study is a long-term, ongoing cardiovascular cohort study of residents of Framingham, Massachusetts. Initiated in 1948, it has significantly contributed to our understanding of cardiovascular disease and its risk factors.

In this project, we:

- Perform exploratory data analysis (EDA) to understand the dataset.
- Preprocess the data, including handling missing values and feature scaling.
- Build and evaluate multiple machine learning models to predict CHD risk.

## 🧾 Dataset

- **Source**: [Kaggle - Framingham Heart Study Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)
- **Instances**: 4,240
- **Features**: 15 attributes including demographic, behavioral, and medical information.
- **Target Variable**: `TenYearCHD` (1 indicates development of CHD within 10 years, 0 otherwise)

## 🔍 Exploratory Data Analysis

The EDA includes:

- Statistical summaries of the features
- Visualization of feature distributions
- Correlation analysis to identify relationships between variables

## 🛠️ Data Preprocessing

Steps undertaken:

- Imputation of missing values
- Encoding categorical variables
- Feature scaling using standardization
- Splitting the data into training and testing sets

## 🤖 Machine Learning Models

Models implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## 📈 Model Evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score


## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Kaggle

## 🚀 Future Work

- Hyperparameter tuning for model optimization
- Implementation of ensemble methods like XGBoost
- Deployment of the model using Flask or Streamlit
- Integration of SHAP values for model interpretability

## 👩‍💻 Author

**Divya Gunde** – Data Analyst | Business Intelligence Analyst | Machine Learning Enthusiast

## 📄 License

This project is licensed under the [MIT License](LICENSE).
