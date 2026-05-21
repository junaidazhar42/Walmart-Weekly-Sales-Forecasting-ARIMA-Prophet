# 🛒 Walmart Weekly Sales Forecasting using ARIMA & Prophet
 
A time series forecasting project that predicts weekly sales across 45 Walmart stores 
using classical statistical models (AR, MA, ARIMA, SARIMA) and Facebook Prophet. The 
project covers the full data science pipeline — from EDA to model evaluation and iterative improvement.

---

## 📁 Dataset
 
**Source:** [Walmart Store Sales Forecasting — Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
 
| Column | Description                                          |
|---|------------------------------------------------------|
| `Store` | Store number (1–45)                                  |
| `Date` | Date (DD-MM-YYYY)                                    |
| `Weekly_Sales` | Sales for the given store                            |
| `Holiday_Flag` | 1 if the week contains a public holiday, 0 otherwise |
| `Temperature` | Temperature on the day of sale                       |
| `Fuel_Price` | Cost of fuel in the region                           |
| `CPI` | Consumer Price Index                                 |
| `Unemployment` | Unemployment rate                                    |
 
- **Total Records:** 6,435 rows × 8 columns
- **Time Period:** February 2010 — October 2012
- **Stores:** 45

---

## 📂 Project Structure
 
```
walmart-weekly-sales/
│
├── main.ipynb                  # Main notebook (EDA + Modelling + Evaluation)
├── Walmart.csv                 # Dataset
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
├── .gitignore                  # Files and folders to be ignored by Git
```
 
---

## ⚙️ Setup & Installation
 
### Prerequisites
- Python 3.8+
- Jupyter Notebook
### Install Dependencies
 
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn prophet
```
 
---

## 🔄 Workflow
 
```
Data Loading → Data Cleaning → EDA → Stationarity Tests → ACF/PACF Analysis → Model Building → Evaluation → Improvements
```
 
---

## 🔍 Exploratory Data Analysis
 
### Key Findings
 
- **Top Performing Store:** Store 20 with **$301.4M** in total sales, followed by Store 4 ($299.5M) and Store 14 ($289.0M).
- **Holiday Impact:** Holiday weeks generate **~7.84% higher** average sales than non-holiday weeks.
- **Trend:** A clear upward sales trend is visible across the 2-year period.
- **Seasonality:** Pronounced annual spikes during late November and December (Thanksgiving & Christmas).
- **Stationarity:** Both ADF and KPSS tests confirm the aggregated weekly series is stationary.

### EDA Plots Covered
- Top 10 stores by total sales (bar chart)
- Weekly sales trend — top 5 stores (multi-line time series)
- Correlation heatmap of all numerical features
- Holiday vs non-holiday sales comparison
- Relationship of different factors with weekly sales (scatter plots)
- Weekly aggregated sales over time for all stores (line chart)

### Other Plots
- Time series decomposition (trend, seasonality, residuals) with `period=52`
- ACF / PACF plots for lag selection
- Line chart to show forecasts of different models 

---

## 🤖 Models
 
All models are trained on aggregated weekly sales across all 45 stores, with the **last 13 weeks held out** as the test set.
 
| Model | Type | Key Parameters |
|---|---|---|
| AR | Baseline | `lags=1` |
| MA | Baseline | `order=(0,0,1)` |
| ARIMA | Intermediate | `order=(6,2,4)` |
| SARIMA | Seasonal | `order=(3,1,3)`, `seasonal_order=(0,1,2,52)` |
| Prophet | Advanced | `seasonality_mode='multiplicative'`, US holidays |
 
---

## 📊 Results
 
### Final Model Comparison (After Improvements)
 
| Rank | Model   | MAE           | RMSE          | MAPE      |
|---|---------|---------------|---------------|-----------|
| 🥇 1 | SARIMA  | $801,024.68   | $996,366.17   | **1.73%** |
| 🥈 2 | Prophet | $929,078.44   | $1,190,522.27 | 1.99%     |
| 3 | ARIMA   | $1,434,128.37 | $1,680,828.45 | 3.13%     |
| 4 | MA      | $1,326,156.81 | $1,745,598.38 | 2.94%     |
| 5 | AR      | $1,387,228.68 | $1,750,302.23 | 3.07%     |

---

## 📌 Conclusion
 
SARIMA (s=52) is the best-performing model for this dataset with a **MAPE of 1.73%**, meaning forecasts are 
off by less than 2% on average. The project demonstrates that correct seasonal period selection is more impactful than model complexity — a simple SARIMA with the right seasonality outperformed a more sophisticated Prophet model.