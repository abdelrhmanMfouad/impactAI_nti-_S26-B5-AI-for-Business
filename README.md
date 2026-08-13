# 🎓 Impact of AI on Students — Post-GPA Prediction

A Machine Learning project that predicts a student's **Post-Semester GPA** based on academic performance, Generative AI usage, study habits, and behavioral factors.

The project covers the complete Machine Learning workflow:

**Data → Preprocessing → Model → Evaluation → Deployment → Decision Support**

---

## 📌 Project Overview

Generative AI is increasingly becoming part of students' academic lives. While AI tools can support learning and productivity, excessive or dependent usage may also affect academic performance, skill retention, and student well-being.

This project aims to investigate these relationships and build a Machine Learning model capable of predicting a student's **Post-Semester GPA** using multiple academic, AI-related, and contextual features.

---

## 🎯 Problem Statement

> **Can we predict a student's future Post-Semester GPA using their academic performance, GenAI usage, study habits, and behavioral factors?**

This is formulated as a **Regression Problem** because the target variable, `Post_Semester_GPA`, is continuous.

---

## 📊 Dataset

The dataset contains:

* **50,000 records**
* **16 features**
* Academic information
* Generative AI usage information
* Study behavior
* Student dependency and anxiety indicators
* Institutional context

### Dataset Features

| Feature                      | Description                                     | Type        |
| ---------------------------- | ----------------------------------------------- | ----------- |
| `Student_ID`                 | Unique student identifier                       | ID          |
| `Major_Category`             | Student's major category                        | Categorical |
| `Year_of_Study`              | Student's academic year                         | Categorical |
| `Pre_Semester_GPA`           | GPA before the semester                         | Numerical   |
| `Weekly_GenAI_Hours`         | Weekly GenAI usage                              | Numerical   |
| `Primary_Use_Case`           | Main purpose of GenAI usage                     | Categorical |
| `Prompt_Engineering_Skill`   | Prompt engineering skill level                  | Categorical |
| `Tool_Diversity`             | Variety of GenAI tools used                     | Numerical   |
| `Paid_Subscription`          | Whether the student uses a paid AI subscription | Boolean     |
| `Traditional_Study_Hours`    | Traditional study hours                         | Numerical   |
| `Perceived_AI_Dependency`    | Perceived dependence on AI                      | Numerical   |
| `Institutional_Policy`       | Institution's AI policy                         | Categorical |
| `Anxiety_Level_During_Exams` | Exam anxiety level                              | Numerical   |
| `Post_Semester_GPA`          | GPA after the semester                          | **Target**  |
| `Skill_Retention_Score`      | Student skill retention score                   | Numerical   |
| `Burnout_Risk_Level`         | Student burnout risk level                      | Categorical |

---

## 🔎 Data Science Workflow

### 1. Data Selection

The dataset was selected because it provides a combination of:

* Academic features
* GenAI usage patterns
* Study behavior
* Psychological/contextual indicators

This makes it suitable for studying the potential relationship between AI usage and student academic outcomes.

### 2. Data Preprocessing

The preprocessing stage includes:

* Handling categorical variables
* Encoding categorical features
* Scaling numerical features when required
* Preparing the dataset for Machine Learning
* Separating features from the target variable

The preprocessing steps are integrated into a **Machine Learning Pipeline** to ensure that the same transformations are applied during both training and deployment.

### 3. Exploratory Data Analysis

EDA was performed to investigate relationships such as:

* GenAI usage vs. GPA
* Pre-Semester GPA vs. Post-Semester GPA
* AI dependency vs. skill retention
* AI usage vs. burnout risk
* Traditional study hours vs. academic performance
* Tool diversity and academic outcomes

---

## 🤖 Machine Learning Models

Several regression models were trained and compared:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* AdaBoost Regressor
* Gradient Boosting Regressor

The models were evaluated using:

* **MAE — Mean Absolute Error**
* **MSE — Mean Squared Error**
* **R² — Coefficient of Determination**

---

## 📈 Model Evaluation

Based on the evaluated results:

| Model                 |     R² Score |
| --------------------- | -----------: |
| **Gradient Boosting** | **0.913941** |
| Random Forest         |     0.902776 |
| Linear Regression     |     0.897209 |
| AdaBoost              |     0.846940 |
| Decision Tree         |     0.802641 |

### 🏆 Best Model

**Gradient Boosting Regressor**

with an **R² score of 0.913941**.

This indicates that the model explains approximately **91.4% of the variance** in the Post-Semester GPA on the evaluated test data.

---

## 🚀 Deployment

The trained Machine Learning pipeline was deployed as an interactive web application using **Streamlit**.

### Application Flow

```text
Student Input
      ↓
Streamlit Web Interface
      ↓
Input Validation
      ↓
Preprocessing Pipeline
      ↓
Trained ML Model
      ↓
Predicted Post-Semester GPA
```

Students can enter their academic and GenAI-related information and receive a predicted Post-Semester GPA.

### 🌐 Live Application

**[Open the Live Streamlit Application](https://impact-ai-nti-s26-b5-ai-for-business.streamlit.app/)**

---

## 🛠️ Technologies Used

| Technology       | Purpose                                    |
| ---------------- | ------------------------------------------ |
| **Python**       | Main programming language                  |
| **Pandas**       | Data manipulation and analysis             |
| **Scikit-learn** | Machine Learning and preprocessing         |
| **Joblib**       | Saving and loading the trained ML pipeline |
| **Streamlit**    | Web application and deployment             |
| **GitHub**       | Source code management                     |

---

## 📁 Project Structure

```text
Impact-of-AI-on-Students/
│
├── app.py
├── impactAI.pkl
├── options.json
├── requirements.txt
├── README.md
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── model_training.ipynb
│
└── images/
    └── model_results.png
```

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Impact-of-AI-on-Students.git
```

Navigate to the project directory:

```bash
cd Impact-of-AI-on-Students
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔮 Future Improvements

Future versions of the project could include:

* Stronger Cross-Validation
* Hyperparameter Optimization
* Feature Importance Analysis
* Explainable AI (SHAP)
* Model Monitoring after deployment
* Continuous retraining with new data
* More advanced regression models

---

## 📌 Key Takeaway

This project demonstrates a complete end-to-end Machine Learning workflow:

> **Data → Analysis → Preprocessing → Modeling → Evaluation → Deployment**

The final product transforms student and GenAI-related information into a **data-driven Post-Semester GPA prediction**, demonstrating how Machine Learning can be integrated into an interactive decision-support application.

---

## 👥 Project

**AI for Business — NTI**

### Project: Impact of AI on Students

Built with ❤️ using Python, Scikit-learn, Joblib, Streamlit, and GitHub.
