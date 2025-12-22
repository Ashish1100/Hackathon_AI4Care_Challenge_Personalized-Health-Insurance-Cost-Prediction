# Hackathon-AI4Care Challenge: Personalized Health Insurance Cost Prediction

## Company Overview
**SecureLife Insurance Brokers** is a digital-first insurance brokerage firm specializing in **personalized health insurance solutions**. The company aims to move beyond traditional actuarial pricing by leveraging data-driven intelligence to deliver accurate, customer-centric insurance recommendations.

---

## Business Problem

SecureLife currently relies on **traditional actuarial tables** and **basic demographic rules** to recommend insurance coverage. This one-size-fits-all approach has resulted in:

- **Under-insurance (35%)**  
  Customers face financial hardship when actual medical expenses exceed coverage.

- **Over-insurance (28%)**  
  Customers overpay for unnecessary coverage, increasing dissatisfaction and churn.

- **Competitive Disadvantage**  
  Data-driven competitors provide more accurate, personalized insurance quotes.

---

## Business Opportunity

By building a **predictive healthcare cost model**, SecureLife can:

- Recommend **optimal, personalized insurance coverage**
- Reduce **claim-to-premium ratios**
- Improve **customer satisfaction and retention**
- Gain a **competitive edge** in the digital insurance market

---

## Objective

As a **Machine Learning Engineer at SecureLife Insurance Brokers**, the objective is to:

> **Predict annual healthcare costs (USD) for individual customers** using demographic, health, and lifestyle data — enabling accurate insurance coverage recommendations.

---

## Problem Statement

### Goal
Predict the **annual medical expenses (USD)** for each customer using an AI/ML regression model.

### Evaluation Metric
- **RMSE (Root Mean Squared Error)**  
  Lower RMSE indicates better predictive performance.

---

## Dataset Description

The dataset is divided into **Train** and **Test** sets.

### Train Dataset (`train.csv`)
Contains labeled medical costs and is used for model training and validation.

**Columns:**
- `age` – Age of the customer
- `sex` – Gender of the customer
- `bmi` – Body Mass Index
- `children` – Number of children
- `smoker` – Smoking status (yes/no)
- `region` – Residential region
- `charges` – Annual medical cost (USD) **[Target Variable]**

---

### Test Dataset (`test.csv`)
Contains **unlabeled data** used for final prediction.

**Columns:**
- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`

---

## Methodology

The project follows a **structured machine learning pipeline**:

### 1. Data Understanding & Exploration
- Examined distributions of age, BMI, smoking status, and charges
- Identified strong cost drivers (e.g., smoking, BMI, age)
- Analyzed outliers and skewness in medical costs

### 2. Data Preprocessing
- Encoded categorical variables (`sex`, `smoker`, `region`)
- Handled skewed target distribution
- Feature scaling where required
- Ensured train–test consistency

### 3. Feature Engineering
- Created interaction awareness (e.g., smoker × BMI)
- Separated smoker vs non-smoker behavioral patterns
- Validated feature importance using tree-based models

### 4. Model Building
- Baseline regression models for benchmarking
- Advanced ensemble model using **XGBoost Regressor**
- Tuned hyperparameters (depth, learning rate, estimators)
- Applied **cross-validation** to avoid overfitting

### 5. Model Evaluation
- Evaluated using **RMSE and R²**
- Compared training vs validation performance
- Checked generalization stability

### 6. Prediction & Submission
- Generated predictions for test dataset
- Post-processed outputs (non-negative constraints)
- Created final submission file as per hackathon format

---

## Tools & Technologies

### Programming & Environment
- **Python 3.x**
- **Jupyter Notebook**

### Libraries
- **NumPy** – Numerical computation
- **Pandas** – Data manipulation
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn** – Preprocessing, metrics, validation
- **XGBoost** – High-performance gradient boosting

### Version Control
- **Git & GitHub**

---

## Model Performance (Summary)

- Strong predictive accuracy on training data
- Stable cross-validation RMSE
- Captures non-linear relationships effectively
- Robust handling of high-cost medical outliers

*(Exact metrics available in the notebook and HTML report)*

---

## Submission Format

The final submission file:
- Format: `.csv`
- Rows: **268 predictions**
- Columns:
  - `customer_id` – Unique identifier
  - `charges` – Predicted annual medical cost (USD)

Example:
```csv
customer_id,charges
1,13452.32
2,4567.89
```
---

## Project Structure

```text
Personalized_Health_Insurance_Cost_Prediction/
│
├── notebook/
│   ├── Version_1_Personalized_Health_Insurance_Cost_Prediction.ipynb   # Main Jupyter Notebook
│   ├── Version_1_Personalized_Health_Insurance_Cost_Prediction.html    # HTML export of the notebook
│   └── notebook.txt                                                    # Notebook-related notes
|
├── data/
│   ├── Train_data__Insurance.csv                                       # Training dataset
│   ├── Test_data_Insurance.csv                                         # Test dataset
│   └── data.txt                                                        # Data-related notes
│
├── output/
│   ├── securelife_cost_predictions_v1.csv                              # Final model predictions
│   └── prediction.txt                                                  # Prediction summary notes
│
├── README.md                                                           # Project documentation
│
└── requirements.txt                                                    # Python dependencies


```

---
## Author

- **Name**: *Ashish Saha*
- **Role**: Data Science & Artificial Intelligence
- **Email**: [ashishsaha.software@gmail.com](mailto:ashishsaha.software@gmail.com)
- **LinkedIn**: [linkedin.com/in/ashishsaha21](https://www.linkedin.com/in/ashishsaha21)
- **GitHub**: [github.com/Ashish1100](https://github.com/Ashish1100)

---

## Contributing
Feel free to **fork**, **adapt**, and **extend** this project for further analysis.

---

## License
> This project is a personal academic initiative developed for **educational purposes and non-commercial** use only.
