# CardioGuard-Protecting hearts with AI
![CardioGuard-Protecting hearts with AI](https://github.com/user-attachments/assets/06ee84df-1873-4089-b19f-f18236335932)

CardioGuard – Protecting Hearts with AI is a machine learning-based system designed to predict heart disease based on an individual's lifestyle. We utilize **logistic regression** as our predictive model since it provides a binary output (0 for no disease, 1 for disease), making it ideal for classification tasks.

**About The Dataset**

The model is trained on the Heart Disease dataset from Kaggle, which contains 1,191 records and 12 features. We use logistic regression for prediction, as it provides a binary classification (0 for no disease, 1 for disease).

The dataset consists of 12 key attributes:

Age – Age of the individual in years.

Sex – Gender (0 = Female, 1 = Male).

Chest Pain Type – Type of chest pain (0-3, indicating severity).

Resting Blood Pressure – Blood pressure measured at rest (mm Hg).

Cholesterol – Serum cholesterol level (mg/dL).

Fasting Blood Sugar – Whether fasting blood sugar is >120 mg/dL (1 = Yes, 0 = No).

Resting ECG – Results of resting electrocardiographic test (0-2).

Max Heart Rate – Maximum heart rate achieved during exercise.

Exercise-Induced Angina – Whether exercise-induced angina occurred (1 = Yes, 0 = No).

Oldpeak – ST depression induced by exercise relative to rest.

ST Slope – The slope of the peak exercise ST segment (0-2).

Target – The dependent variable (0 = No Heart Disease, 1 = Heart Disease).

Using this dataset, CardioGuard enables early detection of heart disease, allowing users to assess their risk based on their health parameters.

Click Here to use the dataset:- https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final/data


**Steps to Create CardioGuard Prediction  App**

requirements.txt – This file contains the required libraries for building the CardioGuard prediction app. You can install them using pip install -r requirements.txt.

basic_eda.py – This script helps in exploratory data analysis (EDA), allowing you to understand the dataset, how the data is structured, and how input and output values are stored.

accuracy.py – This script evaluates the model’s accuracy, checking how well it predicts heart disease.

ui.py – This is the main part of the app, where the user interface (UI) is built. It connects the UI with the machine learning model, allowing users to input values and get predictions.

