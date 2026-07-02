# Quik SKU Success Predictor

Pre-listing SKU success prediction for Quik (Careem's quick-commerce). Given a new SKU's disposition sheet, the model predicts whether it will succeed commercially **before** it goes live — using only signals available at listing time.

---

## Repository Contents

```
quik-sku-success-predictor/
├── app/
│   └── app.py                            # Streamlit prediction app
├── notebooks/
│   └── quik_lr_final_model.ipynb         # Final LR model + score_skus() function
├── reports/
│   ├── final_4way_with_importances.pdf   # Model comparison + feature importances
│   ├── final_4way_full_metrics.pdf       # Full metrics table (Acc/Prec/Rec/F1/AUC)
│   └── lr_coefficient_consistency.pdf    # LR coefficient sign-stability across folds
├── requirements.txt
└── README.md
```

---

## Model Comparison

Four variants are benchmarked across 5 months (Jul25–Dec25) using **Leave-One-Month-Out (LOMO)** evaluation:

| Model | Signals used | LOMO Accuracy | Std |
|---|---|---|---|
| Rule-based (TP CQ) | Pre-listing scores | ~65% | — |
| Logistic Regression | Pre-listing only | **73.1%** | ±2.0% |
| Gradient Boosting | Pre-listing only | 71.8% | ±6.8% |
| Gradient Boosting + Availability | Pre-listing + Availability | ~75% | ±4.1% |

### Why not use Availability?

Availability is a **post-listing signal** — it measures how consistently a SKU stays in stock after it goes live, which is not known at pre-listing time. Including it leaks future information and makes the model undeployable in production.

The GBM+Availability model is included in `reports/` as a reference ceiling. The deployed app uses **LR / RF / GBM without availability**.

### Why Logistic Regression?

- Lowest variance across months (±2.0% vs GBM's ±6.8%)
- GBM collapses on Sep25 (51.8% — near random)
- All top-15 LR feature coefficients maintain the **same sign** across all 5 LOMO folds — evidence of stable, real learned patterns rather than overfitting
- See `reports/lr_coefficient_consistency.pdf` for the full sign-stability analysis

---

## Label Definition

```
Success = 1  if  max(ROS_m1, ROS_m2) > max(min(Bucket_P10_Peer_ROS, Vel_Floor), 1)
```

Training filter: `Availability > 70%` (the official TP CQ listing gate) applied to historical backtest data only.

## Feature Set (30 features)

| Group | Features |
|---|---|
| Raw scores (12) | CQ, Velocity, Benchmark, Satisfaction, Launch, NMI Unit%, NMI SKU%, Channel, Return, Monitoring, Supplier, Launch% |
| Engineered (16) | vel_ratio, vel_ratio², vel_gap, above_floor, launch_pct, nmi_total, nmi_ratio, sat×vel, cq×launch, cq², rtv×mon, bench×sat, is_width, conc_enc, asp_enc, rtv_binary |
| Enriched (2) | asp_log, subcat_p50mv (from Reference Data sheet) |

**Excluded:** Availability, ROS_m1, ROS_m2 — future/post-listing signals.

---

## Streamlit App

### Setup

```bash
pip install -r requirements.txt
```

Update `BACKTEST_PATHS` in `app/app.py` to point to your local backtest files:

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

### Auto-detected Modes

| Mode | Triggered by | Shows |
|---|---|---|
| **Real-time** | No `Success` column in uploaded sheet | Predictions, probabilities, confidence, signal breakdown |
| **Backtest / Validation** | `Success` column present (0/1 values) | + Accuracy, confusion matrices, rule-based comparison |

### Input Format

- Excel file with `Main Working` sheet
- Headers on row 5 (index 4) — rows 6 and 7 also auto-detected
- `Refrence Data` sheet (optional) — used for Subcat P50 MV lookup
- No `Success` or `TP CQ` columns required for real-time use

### Sidebar Controls

| Control | Default | Purpose |
|---|---|---|
| Min Availability % slider | 0 (off) | Filter SKUs below a given availability threshold |
| Decision threshold | 0.50 | Probability cutoff for Success / Fail label |
| Show engineered features | Off | Inspect the 30 engineered features per SKU |
