# Quik SKU Success Predictor

Pre-listing SKU success prediction for Quik (Careem's quick-commerce). Given a disposition sheet, the model predicts whether a SKU will succeed commercially **before** it goes live — using only signals available at listing time (no ROS, no Availability).

---

## Repository Contents

```
quik-sku-success-predictor/
├── app/
│   └── app.py                          # Streamlit prediction app
├── notebooks/
│   └── quik_lr_final_model.ipynb       # Final LR model + score_skus() production function
├── reports/
│   ├── final_4way_full_metrics.pdf     # Bar chart: Accuracy/Precision/Recall/F1/AUC/ROC per month
│   ├── final_4way_linechart.pdf        # Line chart: same 6 metrics — trends across months
│   ├── final_4way_roc_pr.pdf           # OOF ROC + Precision-Recall curves (all 4 models)
│   └── lr_coefficient_consistency.pdf  # LR coefficient sign-stability + top-15 features
├── requirements.txt
└── README.md
```

---

## Model Comparison

All models evaluated with **Leave-One-Month-Out (LOMO)** across 5 months (Jul25–Dec25).  
Training filter: `Availability > 70%` (official TP CQ listing gate). N = 791 SKUs.

| Model | Signals | Acc | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Rule-based (TP CQ) | Final Score ≥ 50 | ~0.65 | ~0.58 | ~0.66 | ~0.59 | ~0.65 |
| **LR No-Avail** ✅ | Pre-listing only | **0.69** | **0.64** | **0.73** | **0.67** | **0.75** |
| GBM No-Avail | Pre-listing only | 0.68 | 0.62 | 0.66 | 0.65 | 0.71 |
| GBM +Avail | Pre-listing + Availability | ~0.78 | ~0.71 | ~0.73 | ~0.67 | ~0.82 |

### Why LR is the deployed model
- Most **consistent** across months (±2.0% std vs GBM's ±6.8%)
- GBM collapses on Sep25 (51.8% — near random)
- All top-15 coefficients maintain the **same sign** across all 5 LOMO folds
- See `reports/lr_coefficient_consistency.pdf`

### Why Availability is excluded from the app
`Availability` is a post-listing signal — it reflects how consistently a SKU stays in stock **after** it goes live. Unknown at pre-listing time. `GBM +Avail` is kept in the reports as a reference ceiling only.

---

## Label & Feature Set

**Success label:**
```
Success = 1  if  max(ROS_m1, ROS_m2) > max(min(Bucket_P10_Peer_ROS, Vel_Floor), 1)
```

**30 features (no Availability, no ROS):**

| Group | Features |
|---|---|
| Raw scores (12) | CQ, Velocity, Benchmark, Satisfaction, Launch%, NMI Unit%, NMI SKU%, Channel, Return, Monitoring, Supplier, Launch Score |
| Engineered (16) | vel_ratio, vel_ratio², vel_gap, above_floor, launch_pct, nmi_total, nmi_ratio, sat×vel, cq×launch, cq², rtv×mon, bench×sat, is_width, conc_enc, asp_enc, rtv_binary |
| Enriched (2) | asp_log, subcat_p50mv (from Reference Data sheet) |

---

## Streamlit App

### Setup

```bash
pip install -r requirements.txt
```

Update `BACKTEST_PATHS` in `app/app.py` to your local backtest file paths:

```python
BACKTEST_PATHS = {
    "Jul25": "/path/to/Jul'25 Backtest.xlsx",
    ...
}
```

```bash
streamlit run app/app.py
```

### Controls

| Control | Default | Description |
|---|---|---|
| Min Availability % | 0 (off) | Filter input SKUs below this threshold before scoring |
| Global threshold | 0.50 | Probability cutoff applied to all models |
| Per-model overrides | 0.00 (uses global) | Fine-tune threshold per model: LR / RF / GBM separately |
| Show engineered features | Off | Inspect the 30 feature values per SKU |

**Per-model thresholds:** expand the "🔧 Per-model threshold overrides" section in the sidebar. Set to `0.00` to inherit the global threshold.

### Auto-detected Modes

| Mode | Triggered by | Shows |
|---|---|---|
| **Real-time** | No `Success` column | Predictions, probabilities, confidence, signal breakdown |
| **Backtest / Validation** | `Success` column present (0/1) | + Accuracy, confusion matrices, rule-based comparison |

### Input Format
- Excel file with `Main Working` sheet (headers on row 5 — or 6/7, auto-detected)
- `Refrence Data` sheet (optional) for Subcat P50 MV lookups
- No `Success` or `TP CQ` columns required for real-time scoring
