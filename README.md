# 📦 Shipment Delay Risk Predictor

An end-to-end machine learning tool that predicts the risk of a shipment being delivered late, 
and explains *why* — built on 180,000+ real supply chain records, with a focus on logistics 
operations in the Middle East and globally.

🔗 **[Live App](your-streamlit-link-goes-here)**

## Business Problem

Late deliveries are one of the most expensive, reputation-damaging problems in logistics. 
This project set out to answer: **can we predict which shipments are at risk of being late, 
before they happen — and can we explain why, so operations teams can act on it?**

## Key Findings

- Expedited shipping tiers (First Class, Second Class) systematically **over-promise delivery 
  times by roughly 2x** — e.g. First Class promises 1 day but actually averages 2. Standard 
  Class is the only tier that reliably meets its own promise.
- Late delivery risk is consistently high (53-58%) **across every region**, including West Asia 
  (Middle East) at 55.3% — confirming this is a company-wide scheduling issue, not a 
  region-specific logistics problem.
- The two features `Days for shipment (scheduled)` and `Shipping Mode` account for over 
  **80% of the model's predictive power** — pointing to a clear, actionable fix: recalibrate 
  delivery promises on expedited tiers.

## Model Performance

Trained a Random Forest Classifier, carefully validated to avoid data leakage 
(see `notebooks/` for the full process, including a leakage bug I caught and fixed):

| Metric | Score |
|---|---|
| Precision | 80.5% |
| Recall | 58.4% |
| ROC-AUC | 0.734 |

**Precision was prioritized over recall** — in a real deployment, a high-precision model means 
operations teams can trust a "high risk" flag and act on it confidently, rather than chasing 
false alarms.

## Features

- Interactive risk prediction based on shipment details (shipping mode, region, pricing, discount)
- Live SHAP-based explanation showing exactly which factors drove each individual prediction
- Built on real transactional data, not synthetic data

## Tech Stack

Python · Pandas · Scikit-learn · SHAP · Streamlit · Matplotlib/Seaborn

## Project Structure

shipment-delay-predictor/
├── data/               # Raw and cleaned datasets
├── notebooks/          # EDA, feature engineering, model training, SHAP analysis
├── app/                # Streamlit app + saved model files
└── README.md

## Run Locally

```bash
git clone https://github.com/bilaljaleel-ux/shipment-delay-predictor.git
cd shipment-delay-predictor
pip install -r requirements.txt
cd app
streamlit run app.py
```

## Known Limitations

- Some engineered features (order month, day of week) showed minimal predictive value and 
  are included for completeness rather than impact.
- The model assumes realistic shipping mode + scheduled day combinations; unrealistic 
  combinations (e.g. Standard Class promised in 1 day) can produce less reliable predictions.

## Author

**Bilal Jaleel** — Data Analyst / Data Scientist  
[LinkedIn]www.linkedin.com/in/bilal-jaleel-ds · [GitHub]https://github.com/bilaljaleel-ux