# Credit Risk Prediction (Loan Default Prediction) Project

## Project Overview
The Credit Risk Prediction Project aims to predict whether a loan applicant will **default** (1) or **not default** (0) based on their personal and financial information. By analyzing historical loan data, the model helps financial institutions make **better-informed decisions** while reducing risk exposure.

---

## Problem Statement
Loan default is a major risk for financial institutions. This project seeks to answer:  
- **Which applicants are likely to default?**  
- **Which features (income, employment, dependents, education, etc.) influence default the most?**  

**Objective:** Build a **machine learning model** to classify applicants as defaulters or non-defaulters, providing actionable insights for credit evaluation.

---

## Dataset
The dataset used in this project is sourced from **Kaggle**:  
**[Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)**  

**Columns include:**  

| Column | Description |
|--------|-------------|
| LoanID | Unique identifier for each loan applicant |
| Age | Age of applicant |
| Income | Monthly income |
| LoanAmount | Amount of loan applied |
| MonthsEmployed | Number of months employed |
| HasDependents | Whether applicant has dependents |
| Education | Education level |
| MaritalStatus | Marital status |
| Gender | Applicant gender |
| EmploymentType | Employment type |
| Default | Target variable (0 = No default, 1 = Default) |

---

## Libraries Used
- **Pandas** – Data loading, cleaning, and manipulation  
- **NumPy** – Numerical operations  
- **Matplotlib / Seaborn** – Data visualization and plots  
- **Scikit-Learn** – Machine learning (Logistic Regression), preprocessing, train-test split, evaluation  
- **Joblib** – Saving and loading trained models and preprocessing objects  

---

## Project Steps

### 1. Data Loading and Inspection
- Loaded the dataset into a Pandas DataFrame  
- Checked **shape, info, and missing values**  
- Converted relevant numeric fields to proper types  

### 2. Exploratory Data Analysis (EDA)
- Visualized **distribution of defaulters vs non-defaulters**  
- Analyzed **defaulters by age groups, income, education, marital status, and dependents**  
- Plotted **boxplots, count plots, barplots** for categorical and numeric variables  
- Observed insights such as:
  - Higher default rates among applicants with low income  
  - Dependents and lack of employment stability increase default probability  

### 3. Data Preprocessing
- **Categorical encoding**: Used `LabelEncoder` for categorical features  
- **Scaling numeric features**: StandardScaler to normalize numeric columns  
- **Feature selection**: Removed irrelevant columns like `LoanID`  

### 4. Model Building
- **Train-test split**: 80% training, 20% testing  
- **Model**: Logistic Regression  
- **Evaluation metrics**:
  - Accuracy Score  
  - Confusion Matrix  
  - Classification Report (precision, recall, F1-score)  

### 5. Model Evaluation
- Achieved satisfactory classification accuracy  
- Confusion matrix showed the model correctly predicted a high percentage of both defaulters and non-defaulters  

### 6. Model Saving
- Saved the **trained model** using `joblib` for future predictions  

---

## Key Insights
- Applicants with **low income** or **unstable employment** have a higher probability of default  
- **Education and marital status** influence default patterns  
- Majority of loans were given to applicants with **no dependents**, yet defaulters were concentrated among those with **dependents**  
- Visualizations such as **boxplots, barplots, and countplots** provide a clear view of risk patterns  

---

## Project Structure
Credit-Risk-Prediction/
│
├── Loan_Default_Prediction.ipynb # Jupyter Notebook containing full workflow
├── Loan_Default.csv # Dataset used
├── loan_model.joblib
└── README.md # Project documentation



---

## How to Run
1. Open **Jupyter Notebook**  
2. Load `Loan_Default_Prediction.ipynb`  
3. Run all cells sequentially  
4. Explore **EDA, model building, and evaluation**  

> Optional: Use saved joblib files to **predict on new applicant data** without re-training.

---

## Future Scope
- Build a **fully interactive GUI** using **Streamlit** or **Dash**  
- Include **feature importance analysis** for executives  
- Integrate a **dashboard for real-time predictions** on new applicants  
- Experiment with **advanced models**: Random Forest, XGBoost, or Neural Networks  

---

## Conclusion
This project demonstrates the **end-to-end workflow of a machine learning model** for credit risk prediction: from data cleaning, EDA, preprocessing, model training, evaluation, to saving the trained model. It provides a **practical tool for financial institutions** to reduce loan default risk.

---

## Author
**Isbah Ali** – Data Analytics Intern
