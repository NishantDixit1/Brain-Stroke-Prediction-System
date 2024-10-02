# Brain-Stroke-Prediction-System
The Brain Stroke Prediction System uses machine learning to predict stroke risk based on health factors like age, hypertension, heart disease, and more. This project preprocesses data, trains models, and evaluates them using accuracy, precision, and recall, aiming to assist in early stroke detection.
# Problem Statement
Stroke is a severe medical condition that can cause lasting brain damage, disability, and even death. The primary challenge is the early prediction of stroke occurrences in high-risk individuals to allow timely medical intervention. Since stroke is caused by a combination of several risk factors like age, hypertension, heart disease, and more, predicting its occurrence accurately using machine learning can provide a significant advantage in preventing adverse outcomes.

# Objective
The objective of this project is to develop a machine learning model capable of predicting the likelihood of a person having a stroke based on specific input parameters like:

1.Age
2.Gender
3.Hypertension
4.Heart Disease
5.Smoking Status
6.BMI
7.Glucose Levels

# Features

1.Data Preprocessing: Cleaning and transforming the raw dataset to handle missing values and categorical data.
2.Feature Engineering: Selecting the most relevant features for accurate stroke prediction.
3.Model Training: Implementing various machine learning algorithms such as Logistic Regression, Random Forest, and Decision Trees to determine the best-performing model.
4.Model Evaluation: Using evaluation metrics like Accuracy, Precision, Recall, and F1-Score to assess the performance of the models.
5.Prediction: The trained model predicts whether a person is likely to experience a stroke based on input features.
Technologies Used
6.Python: The core programming language used for data analysis and model development.
7.Machine Learning Libraries: Scikit-learn, TensorFlow, or Keras for building predictive models.
8.Pandas & NumPy: For data manipulation and preprocessing.
9.Matplotlib & Seaborn: For data visualization and analysis of features.
10.Jupyter Notebook: For development and experimentation with the models.

# Dataset
The dataset used in this project includes health-related attributes that contribute to stroke prediction. The dataset contains information such as age, gender, hypertension status, heart disease history, smoking status, BMI, and glucose levels. The dataset can be found at Kaggle - Stroke Prediction Dataset.

# Machine Learning Models
1.Logistic Regression: A basic classification model used as a baseline.
2.Random Forest: A more complex model that builds multiple decision trees and averages the results for better accuracy.
3.Support Vector Machine (SVM): Used for separating data into categories with a decision boundary.
4.K-Nearest Neighbors (KNN): A simple, instance-based learning model.
5.XGBoost: An advanced implementation of gradient boosting for improved accuracy.

# Results
The models are evaluated based on the following metrics:

Accuracy: The percentage of correct predictions made by the model.
Precision: The ratio of true positive predictions to the total predicted positives.
Recall: The ratio of true positives to the total actual positives.
F1 Score: The harmonic mean of Precision and Recall, balancing both metrics.
Conclusion
This project demonstrates how machine learning models can be leveraged to predict the risk of stroke in patients, potentially aiding healthcare professionals in early diagnosis and prevention. By analyzing key health parameters, the Brain Stroke Prediction System aims to help in reducing the mortality rate and improving treatment outcomes for stroke patients.


Future Enhancements
Incorporating more real-world datasets to improve accuracy.
Developing a user-friendly web interface for real-time stroke prediction.
Integrating the system with healthcare platforms for broader accessibility.
