# Quantitative-Research

**A collection of advanced quantitative finance and risk modeling tasks.**

---

## 📚 Overview
This repository contains a suite of Python-based solutions for real-world quantitative research and risk management problems, including:

- **Natural Gas Price Estimation & Storage Contract Valuation**
- **Loan Default Probability (PD) Modeling & Expected Loss Calculation**
- **FICO Score Quantization & Rating Map Optimization**

Each module is designed for clarity, extensibility, and practical use in a professional or academic setting.

---

## 🚀 Project Structure

- `gas_price_analysis.py`  
  _Estimate natural gas prices for any date and value storage contracts with custom cash flows._
- `loan_default_analysis.py`  
  _Predict loan default probability (PD) and expected loss using machine learning._
- `task4.py`  
  _FICO score bucketing and quantization using dynamic programming and log-likelihood/MSE optimization._
- `Task 3 and 4_Loan_Data.csv`  
  _Loan borrower dataset for PD modeling and quantization._
- `Nat_Gas.csv`  
  _Monthly natural gas price data for time series modeling._
- `requirements.txt`  
  _All required Python dependencies._

---

## 🏆 Key Features

### 1. **Natural Gas Price Estimation & Storage Contract Valuation**
- **Interpolate and extrapolate** gas prices for any date (past or future)
- **Value storage contracts** with custom injection/withdrawal schedules, storage costs, and volume constraints
- **Detailed cash flow breakdown** and storage level tracking

### 2. **Loan Default Probability & Expected Loss Modeling**
- **Train and compare** Logistic Regression and Random Forest models
- **Predict PD** for any borrower profile
- **Calculate expected loss** using a configurable recovery rate
- **Feature importance** visualization for model interpretability

### 3. **FICO Score Quantization & Rating Map**
- **Optimal bucketing** of FICO scores using dynamic programming
- **Supports both log-likelihood and MSE** as optimization criteria
- **Maps FICO scores to ratings** for robust credit risk segmentation

---

## 📝 Example Outputs

### 🔹 Natural Gas Storage Contract
```
Contract Value: $-5,995.53

Cash Flows:
2023-01-01: injection    $-23,232.26
2023-02-01: storage_cost $-930.00
... (truncated)
```

### 🔹 Loan Default Analysis
```
Model Performance:
Logistic Regression ROC AUC: 1.0000
Random Forest ROC AUC: 0.9997

Example Loan Analysis:
Loan Amount: $10,000.00
Probability of Default: 2.13%
Expected Loss: $1,917.00
```

### 🔹 FICO Score Bucketing
```
Optimal FICO boundaries: [545, 602, 650, 700, 750]

Sample mapping:
   fico_score  rating
0         605       2
1         720       5
...
```

---

## 🛠️ How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run any module:**
   ```bash
   python gas_price_analysis.py
   python loan_default_analysis.py
   python task4.py
   ```

---

## 📖 References & Further Reading
- [Quantization (signal processing)](https://en.wikipedia.org/wiki/Quantization_(signal_processing))
- [Likelihood function](https://en.wikipedia.org/wiki/Likelihood_function)
- [Dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming#Computer_programming)

---

## 💡 Contributing
Pull requests and suggestions are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## © 2025 RichieOnData
