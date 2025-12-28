# Hackathon-AI4Care Challenge: Personalized Health Insurance Cost Prediction
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)
![Machine Learning](https://img.shields.io/badge/Type-Regression-success.svg)
![RMSE](https://img.shields.io/badge/Best%20RMSE-4264.12-informational.svg)
![Rank](https://img.shields.io/badge/Leaderboard-Rank%205-brightgreen.svg)
![Hackathon](https://img.shields.io/badge/Hackathon-AI4Care-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

### Hackathon Overview
**Hackathon Name:** Insurance Amount Prediction (AI4Care Challenge)    
**Organizer / Platform:** Great Learning (SecureLife Insurance Brokers)  
**Duration:** 24-hours  
**Team Size:** Individual (Solo Participation)  

This hackathon focused on building a **machine learning model to predict annual medical insurance costs** using customer demographic, health, and lifestyle data. The evaluation was based on minimizing **RMSE** on a hidden test dataset.

---

### Leaderboard Performance

- **Team Name:** BitDecoder  
- **Final Rank:** **🏅 5th Place**  
- **Best RMSE Achieved:** **4264.12**  
- **Submission Type:** Regression (Insurance Cost Prediction)

---

### Top Leaderboard Snapshot

| Rank | Team Name                | Least RMSE |
|-----:|--------------------------|-----------:|
| 1    | SKM     | 4195.48    | 11          |
| 2    | CB   | 4196.70    | 15          |
| 3    | Precision               | 4221.77    |
| 4    | CodeMonkey              | 4223.94    | 
| **5**| **BitDecoder (Me)**     | **4264.12**|


---

### Learnings from the Hackathon

- Smoking status and BMI dominate healthcare cost prediction
- Tree-based ensemble models outperform linear models significantly
- Cross-validation stability is as important as leaderboard score
- Feature interactions matter more than raw feature scaling

---

### Impact

This result validates the effectiveness of a **data-driven underwriting approach** and demonstrates how machine learning can replace traditional actuarial heuristics with **personalized, scalable insurance pricing models**.

---

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

### Train Dataset
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

### Test Dataset
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
## **Author**

<div align="center">

### **Ashish Saha**
**Machine Learning Research** | **AI Engineering** | **Data Science**

*Specializing in building intelligent ML systems and transforming data into actionable insights.*

**Tech Stack:** Python • TensorFlow/Keras • PyTorch • XGBoost • Scikit-learn 

<a href="https://github.com/Ashish1100" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub">
</a>
<a href="https://www.linkedin.com/in/ashishsaha21/" target="_blank">
  <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn">
</a>
<a href="mailto:ashishsaha.software@gmail.com">
  <img src="https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white" alt="Email">
</a>

</div>

---

## License
> This project is a personal academic initiative developed for **educational purposes and non-commercial** use only.

<div align="center">

---

### **Star ⭐ this repo if you found this project helpful!**

---

*Made with ❤️ by Ashish Saha*

</div>
