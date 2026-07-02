# Quik SKU Success Predictor

A pre-listing SKU success prediction model for Quik (Careem's quick-commerce). Given a new SKU's disposition sheet scores, the model predicts whether the SKU will be a commercial success **before** it goes live — using only signals available at listing time (no ROS, no availability).

---

## Problem

Quik's existing rule-based gate (TP CQ) flags SKUs as list-worthy but cannot rank or score them. The goal is a model that:
- Predicts success probability at pre-listing stage
- Uses only **pre-listing signals** (no future data leakage)
- Is consistent and interpretable across time

---

## Approach

### Label Definition
```
Success = 1 if max(ROS_m1, ROS_m2) > max(min(Bucket_P10_Peer_ROS, Vel_Floor), 1)
```

### Feature Set (30 features)
- 12 raw score columns (CQ, Velocity, Benchmark, Satisfaction, Launch, NMI, Channel, Return, Monitoring, Supplier)
- 16 engineered interaction features (vel_ratio, above_floor, nmi_total, sat×vel, cq×launch, etc.)
- 2 enriched features (asp_log, subcat_p50mv from Reference Data sheet)

**Excluded by design:** Availability, ROS_m1, ROS_m2 — these are future/post-listing signals.

### Evaluation: Leave-One-Month-Out (LOMO)
Train on 4 months → test on held-out month. 5 folds total (Jul25, Sep25, Oct25, Nov25, Dec25).  
Random k-fold CV is **not** used — it inflates accuracy ~5.7% due to temporal leakage.

---

## Results

| Model | LOMO Accuracy | Std | Notes |
|---|---|---|---|
| Rule-based (TP CQ) | ~65% | — | Baseline |
| Logistic Regression | **73.1%** | ±2.0% | Most consistent |
| Random Forest | 72.4% | ±3.1% | |
| Gradient Boosting | 71.8% | ±6.8% | Collapses Sep25 (51.8%) |

**LR is the recommended model**: lowest variance across months, all top-15 feature coefficients maintain the same sign across all 5 LOMO folds (sign-stability = strong evidence of real learned patterns).

---

## Repository Structure

```
quik-sku-success-predictor/
├── app/
│   └── app.py                          # Streamlit prediction app
├── notebooks/
│   ├── quik_lr_final_model.ipynb       # Final production LR model + score_skus()
│   ├── quik_prelisting_model_clean.ipynb  # Clean LOMO evaluation
│   ├── quik_sku_all_experiments.ipynb  # Full experiments: rule vs ML, 3 models
│   └── quik_sku_model_v2.ipynb         # Earlier v2 iteration
├── reports/
│   ├── final_4way_with_importances.pdf # 4-model comparison + feature importances
│   ├── lr_coefficient_consistency.pdf  # LR coefficient sign-stability across folds
│   ├── final_4way_full_metrics.pdf     # Full metrics table (Acc/Prec/Rec/F1/AUC)
│   ├── final_3way_comparison.pdf       # Rule vs LR vs GBM
│   ├── rule_vs_lr_focused.pdf          # Focused rule vs best model comparison
│   └── quik_sku_model_summary.pdf      # Executive summary
├── requirements.txt
└── README.md
```

---

## Streamlit App

The app trains all 3 models on the 5 historical backtest files at startup, then scores new SKU sheets.

### Setup

```bash
pip install -r requirements.txt
```

Update `BACKTEST_PATHS` in `app/app.py` to point to your local backtest Excel files:

```python
BACKTEST_PATHS = {
    "Jul25": "/path/to/Jul'25 Backtest.xlsx",
    "Sep25": "/path/to/Sep'25 Backtest.xlsx",
    "Oct25": "/path/to/Oct'25 Backtest.xlsx",
    "Nov25": "/path/to/Nov'25 Backtest.xlsx",
    "Dec25": "/path/to/Dec'25 Backtest.xlsx",
}
```

```bash
streamlit run app/app.py
```

### Two Modes (auto-detected)

| Mode | Triggered by | Shows |
|---|---|---|
| **Real-time** | No `Success` column | Predictions, probabilities, confidence, signal breakdown |
| **Backtest** | `Success` column present (0/1) | + Accuracy metrics, confusion matrices, rule comparison |

### Input Format
- Excel file with `Main Working` sheet
- Headers on row 5 (index 4) — or 6/7, auto-detected
- `Refrence Data` sheet (optional) — used for Subcat P50 MV lookups

---

## Key Design Decisions

- **No availability filter on input** by default — real-time sheets may not have Availability populated yet. Slider in sidebar lets you apply a threshold if needed.
- **Ensemble = mean** of LR + RF + GBM probabilities — more stable than any single model.
- **Model Votes (of 3)** column shows how many of the 3 models agree with the ensemble direction.
- Backtest training uses `Availability > 70%` filter (the official TP CQ listing gate) to ensure training labels reflect post-filter outcomes.
