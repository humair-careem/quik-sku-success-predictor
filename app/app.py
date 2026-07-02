import streamlit as st
import pandas as pd
import numpy as np
import io
import copy
import warnings
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quik SKU Success Predictor",
    page_icon="📦",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MONTHS_ORDERED = ["Jul25", "Sep25", "Oct25", "Nov25", "Dec25"]

BACKTEST_PATHS = {
    "Jul25": "/Users/humair.abbas/Downloads/Jul'25 Backtest.xlsx",
    "Sep25": "/Users/humair.abbas/Downloads/Sep'25 Backtest.xlsx",
    "Oct25": "/Users/humair.abbas/Downloads/Oct'25 Backtest.xlsx",
    "Nov25": "/Users/humair.abbas/Downloads/Nov'25 Backtest.xlsx",
    "Dec25": "/Users/humair.abbas/Downloads/Dec'25 Backtest.xlsx",
}

BASE_FEATURES = [
    "Vel\n Score", "Vel Floor", "Benchmark", "Sat\n Score", "Launch\n Score",
    "CQ\n SCORE", "NMI\n Unit%", "NMI\n SKU%", "CH\n SCORE", "Return\n Score",
    "Mon\n Score", "SP\n SCORE", "rtv_binary", "vel_ratio", "vel_ratio_sq",
    "vel_gap", "above_floor", "launch_pct", "nmi_total", "nmi_ratio",
    "sat_x_vel", "cq_x_launch", "cq_sq", "rtv_x_mon", "bench_x_sat",
    "is_width", "conc_enc", "asp_enc", "asp_log", "subcat_p50mv",
]

FEAT_LABELS = {
    "Vel\n Score": "Vel Score", "Vel Floor": "Vel Floor", "Benchmark": "Benchmark",
    "Sat\n Score": "Sat Score", "Launch\n Score": "Launch Score", "CQ\n SCORE": "CQ Score",
    "NMI\n Unit%": "NMI Unit%", "NMI\n SKU%": "NMI SKU%", "CH\n SCORE": "CH Score",
    "Return\n Score": "Return Score", "Mon\n Score": "Mon Score", "SP\n SCORE": "SP Score",
    "rtv_binary": "RTV Binary", "vel_ratio": "Vel Ratio", "vel_ratio_sq": "Vel Ratio²",
    "vel_gap": "Vel Gap", "above_floor": "Above Floor", "launch_pct": "Launch Pct",
    "nmi_total": "NMI Total", "nmi_ratio": "NMI Ratio", "sat_x_vel": "Sat×Vel",
    "cq_x_launch": "CQ×Launch", "cq_sq": "CQ²", "rtv_x_mon": "RTV×Mon",
    "bench_x_sat": "Bench×Sat", "is_width": "Is Width", "conc_enc": "Conc Level",
    "asp_enc": "ASP Bucket", "asp_log": "ASP (log)", "subcat_p50mv": "Subcat P50 MV",
}

MODEL_META = {
    "Logistic Regression": {"color": "#1565C0", "short": "LR"},
    "Random Forest":       {"color": "#2E7D32", "short": "RF"},
    "Gradient Boosting":   {"color": "#C62828", "short": "GBM"},
}

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df, ref_d):
    keys = df.get("Subcat ASP Key", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()

    def vc(col, fb=0.0):
        return np.array([float(ref_d.get(k, {}).get(col, fb) or fb) for k in keys])

    for col in [
        "Vel\n Score", "Vel Floor", "Benchmark", "Sat\n Score", "Launch\n Score",
        "CQ\n SCORE", "NMI\n Unit%", "NMI\n SKU%", "CH\n SCORE", "Return\n Score",
        "Mon\n Score", "SP\n SCORE", "Launch Success\n %",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else 0.0

    df["rtv_binary"]   = (df.get("RTV Terms", pd.Series("", index=df.index)).astype(str).str.lower() == "yes").astype(int)
    vfn                = df["Vel Floor"].replace(0, np.nan)
    df["vel_ratio"]    = (df["Benchmark"] / vfn).clip(0, 5).fillna(0)
    df["vel_ratio_sq"] = df["vel_ratio"] ** 2
    df["vel_gap"]      = (df["Benchmark"] - df["Vel Floor"]).fillna(0)
    df["above_floor"]  = (df["Benchmark"] >= df["Vel Floor"]).astype(int)
    df["launch_pct"]   = df.get("Launch Success\n %", pd.Series(0, index=df.index))
    df["nmi_total"]    = df["NMI\n Unit%"] + df["NMI\n SKU%"]
    df["nmi_ratio"]    = (df["NMI\n Unit%"] / df["NMI\n SKU%"].replace(0, np.nan)).clip(0, 10).fillna(0)
    df["sat_x_vel"]    = df["Sat\n Score"] * df["Vel\n Score"] / 10000
    df["cq_x_launch"]  = df["CQ\n SCORE"] * (df["Launch\n Score"] > 0).astype(int)
    df["cq_sq"]        = (df["CQ\n SCORE"] / 100) ** 2
    df["rtv_x_mon"]    = df["rtv_binary"] * df["Mon\n Score"]
    df["bench_x_sat"]  = df["Benchmark"] * df["Sat\n Score"] / 10000
    df["is_width"]     = (df.get("Width or Depth", pd.Series("Depth", index=df.index)) == "Width").astype(int)
    df["conc_enc"]     = df.get("Conc\n Level", pd.Series("MEDIUM", index=df.index)).map(
                             {"LOW": 0, "MEDIUM": 1, "HIGH": 2}).fillna(1).astype(int)
    df["asp_enc"]      = df.get("ASP\n Bucket", pd.Series("25+", index=df.index)).map(
                             {"0-5": 0, "5-10": 1, "10-15": 2, "15-20": 3, "20-25": 4, "25+": 5}).fillna(5).astype(int)
    df["asp_log"]      = np.log1p(pd.to_numeric(df.get("ASP", 0), errors="coerce").fillna(0).clip(0, 500))
    df["subcat_p50mv"] = vc("Sub Category P50 ROS (Moving)")
    return df


def load_backtest_month(path, month):
    if month == "Nov25":
        raw = pd.read_excel(path, sheet_name="Main Working", header=None)
        h = raw.iloc[6].tolist(); h[0] = "Barcode"
        df = raw.iloc[7:].copy(); df.columns = [str(x).strip() for x in h]
        cnt = Counter(); nc = []
        for c in df.columns:
            cnt[c] += 1; nc.append(f"{c}.dup{cnt[c]-1}" if cnt[c] > 1 else c)
        df.columns = nc
    elif month == "Dec25":
        df = pd.read_excel(path, sheet_name="Main Working", header=5)
        df.columns = [str(c).strip() for c in df.columns]
    else:
        df = pd.read_excel(path, sheet_name="Main Working", header=4)
        df.columns = [str(c).strip() for c in df.columns]

    df["month"]        = month
    df["Success"]      = pd.to_numeric(df.get("Success", np.nan), errors="coerce")
    df["Availability"] = pd.to_numeric(df.get("Availability", np.nan), errors="coerce")

    ref = pd.read_excel(path, sheet_name="Refrence Data", header=0)
    ref.columns = [str(c).strip() for c in ref.columns]
    ref = ref.drop_duplicates("Subcat_ASP Key", keep="first")
    for c in ref.select_dtypes(include="number").columns:
        ref[c] = pd.to_numeric(ref[c], errors="coerce").fillna(0)
    ref_d = ref.set_index("Subcat_ASP Key").to_dict("index")

    df = engineer_features(df, ref_d)
    return df, ref_d


def load_new_sheet(file_bytes):
    """
    Load a real-time or backtest disposition sheet.
    Returns (df, has_labels, missing_score_cols, error).
    has_labels = True if Success column is present and has values.
    """
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet = "Main Working" if "Main Working" in xl.sheet_names else xl.sheet_names[0]

        df = None
        for header_row in [4, 5, 6, 3]:
            try:
                _df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=header_row)
                _df.columns = [str(c).strip() for c in _df.columns]
                # Need at least one score column to be valid
                if any(c in _df.columns for c in ["CQ\n SCORE", "Vel\n Score", "Benchmark"]):
                    df = _df
                    break
            except Exception:
                continue

        if df is None:
            # Fallback: use first sheet, first row as header
            df = pd.read_excel(io.BytesIO(file_bytes), header=0)
            df.columns = [str(c).strip() for c in df.columns]

        # Deduplicate columns
        cnt = Counter(); nc = []
        for c in df.columns:
            cnt[c] += 1; nc.append(f"{c}.dup{cnt[c]-1}" if cnt[c] > 1 else c)
        df.columns = nc

        # Drop fully empty rows
        df = df.dropna(how="all").reset_index(drop=True)

        df["Availability"] = pd.to_numeric(df.get("Availability", np.nan), errors="coerce")

        # Detect if ground truth is available
        has_labels = False
        if "Success" in df.columns:
            success_vals = pd.to_numeric(df["Success"], errors="coerce")
            if success_vals.isin([0, 1]).sum() > 0:
                df["Success"] = success_vals
                has_labels = True

        # Detect rule-based predictions
        has_rule = "TP CQ" in df.columns

        # Load reference data if available
        ref_d = {}
        if "Refrence Data" in xl.sheet_names:
            ref = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Refrence Data", header=0)
            ref.columns = [str(c).strip() for c in ref.columns]
            if "Subcat_ASP Key" in ref.columns:
                ref = ref.drop_duplicates("Subcat_ASP Key", keep="first")
                for c in ref.select_dtypes(include="number").columns:
                    ref[c] = pd.to_numeric(ref[c], errors="coerce").fillna(0)
                ref_d = ref.set_index("Subcat_ASP Key").to_dict("index")

        df = engineer_features(df, ref_d)

        # Which base feature columns are missing (zero-filled)?
        raw_score_cols = [
            "Vel\n Score", "Vel Floor", "Benchmark", "Sat\n Score", "Launch\n Score",
            "CQ\n SCORE", "NMI\n Unit%", "NMI\n SKU%", "CH\n SCORE",
            "Return\n Score", "Mon\n Score", "SP\n SCORE",
        ]
        missing_scores = [c for c in raw_score_cols if c not in df.columns or df[c].sum() == 0]

        return df, has_labels, has_rule, missing_scores, None

    except Exception as e:
        return None, False, False, [], str(e)


# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on historical data…")
def train_models():
    all_dfs = []
    missing = []
    for month, path in BACKTEST_PATHS.items():
        if os.path.exists(path):
            df, _ = load_backtest_month(path, month)
            all_dfs.append(df)
        else:
            missing.append(month)

    if not all_dfs:
        return None, None, f"No backtest files found. Missing: {missing}"

    data = pd.concat(all_dfs, ignore_index=True)
    data = data[data["Success"].isin([0.0, 1.0])].copy()
    data["Success"] = data["Success"].fillna(-1).astype(int)
    data = data[data["Availability"] > 0.70].reset_index(drop=True)

    X = data[BASE_FEATURES].fillna(0)
    y = data["Success"]

    model_defs = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=3000, C=0.1, penalty="l2", solver="lbfgs",
                class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=600, max_depth=4, min_samples_leaf=2,
            max_features="log2", class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.01,
            subsample=0.8, min_samples_leaf=10, random_state=42
        ),
    }

    trained = {}
    for name, model in model_defs.items():
        m = copy.deepcopy(model)
        m.fit(X, y)
        trained[name] = m

    train_info = {
        "n_skus": len(data),
        "n_months": data["month"].nunique(),
        "success_rate": float(y.mean()),
        "missing_months": missing,
    }
    return trained, train_info, None


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(models, df, model_thresholds, global_threshold):
    X = df[BASE_FEATURES].fillna(0)
    probas = {}
    for name, model in models.items():
        probas[name] = model.predict_proba(X)[:, 1]
    ensemble = np.stack(list(probas.values())).mean(axis=0)
    probas["Ensemble"] = ensemble

    preds = {}
    for name, p in probas.items():
        t = model_thresholds.get(name, global_threshold) if name != "Ensemble" else global_threshold
        preds[name] = (p >= t).astype(int)
    return probas, preds


def build_output(df, probas, preds, threshold, model_thresholds, has_labels, has_rule):
    out = df.copy()

    # Identifier columns (shown first)
    id_cols = [c for c in df.columns if any(x in c.lower() for x in
               ["barcode", "sku", "item", "name", "desc", "subcat", "category", "brand", "supplier"])]
    id_cols = id_cols[:6]

    for name in list(MODEL_META.keys()) + ["Ensemble"]:
        out[f"{name} Prob"] = np.round(probas[name], 3)
        t = model_thresholds.get(name, threshold) if name != "Ensemble" else threshold
        label = "Final Prediction" if name == "Ensemble" else f"{MODEL_META[name]['short']} Pred"
        out[label] = np.where(probas[name] >= t, "✅ Success", "❌ Fail")

    p = probas["Ensemble"]
    out["Confidence"] = np.where(
        (p > 0.7) | (p < 0.3), "High",
        np.where((p > 0.6) | (p < 0.4), "Medium", "Low")
    )

    # Model agreement count (how many of 3 models agree with ensemble)
    votes = np.stack([preds[n] for n in MODEL_META]).sum(axis=0)
    out["Model Votes (of 3)"] = votes

    # Rule-based column if present
    if has_rule:
        tpcq = df["TP CQ"].astype(str).str.strip().str.upper()
        out["Rule Pred"] = np.where(tpcq.isin(["TP", "FP"]), "✅ Success", "❌ Fail")

    return out, id_cols


# ── Accuracy analysis (backtest mode) ────────────────────────────────────────
def render_accuracy_analysis(df, probas, preds, threshold, has_rule):
    valid = df["Success"].isin([0, 1, 0.0, 1.0])
    df = df[valid].reset_index(drop=True)
    y_true = df["Success"].astype(int)
    probas = {k: v[valid.values] for k, v in probas.items()}
    preds  = {k: (v[valid.values] if hasattr(v, '__len__') else v) for k, v in preds.items()}
    ens_pred = preds["Ensemble"]

    st.subheader("📊 Accuracy Analysis (ground truth available)")

    # Per-model metrics table
    rows = []
    for name in list(MODEL_META.keys()) + ["Ensemble"]:
        p = preds[name]
        rows.append({
            "Model": name,
            "Accuracy":  round(accuracy_score(y_true, p), 3),
            "Precision": round(precision_score(y_true, p, zero_division=0), 3),
            "Recall":    round(recall_score(y_true, p, zero_division=0), 3),
            "F1":        round(f1_score(y_true, p, zero_division=0), 3),
            "AUC":       round(roc_auc_score(y_true, probas[name]), 3),
        })
    if has_rule:
        tpcq = df["TP CQ"].astype(str).str.strip().str.upper()
        rule_pred = tpcq.isin(["TP", "FP"]).astype(int)
        rows.insert(0, {
            "Model": "Rule-based (TP CQ)",
            "Accuracy":  round(accuracy_score(y_true, rule_pred), 3),
            "Precision": round(precision_score(y_true, rule_pred, zero_division=0), 3),
            "Recall":    round(recall_score(y_true, rule_pred, zero_division=0), 3),
            "F1":        round(f1_score(y_true, rule_pred, zero_division=0), 3),
            "AUC":       round(roc_auc_score(y_true, rule_pred), 3),
        })

    metrics_df = pd.DataFrame(rows).set_index("Model")
    best_acc = metrics_df["Accuracy"].max()

    def highlight_best(row):
        styles = []
        for col in row.index:
            if row[col] == metrics_df[col].max():
                styles.append("background-color: #C8E6C9; font-weight: bold")
            else:
                styles.append("")
        return styles

    st.dataframe(
        metrics_df.style
            .apply(highlight_best, axis=1)
            .background_gradient(subset=["AUC"], cmap="Blues"),
        use_container_width=True
    )

    # Confusion matrices side-by-side
    models_to_show = (["Rule-based (TP CQ)"] if has_rule else []) + ["Logistic Regression", "Ensemble"]
    fig, axes = plt.subplots(1, len(models_to_show), figsize=(4 * len(models_to_show), 4))
    if len(models_to_show) == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=11)

    for ax, name in zip(axes, models_to_show):
        if name == "Rule-based (TP CQ)":
            tpcq = df["TP CQ"].astype(str).str.strip().str.upper()
            yp = tpcq.isin(["TP", "FP"]).astype(int).values
            col = "#E65100"
        else:
            yp = preds[name]
            col = MODEL_META.get(name, {}).get("color", "#555555")

        cm = confusion_matrix(y_true, yp, labels=[0, 1])
        vmax = cm.max()
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > vmax * 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: Fail", "Pred: Success"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Act: Fail", "Act: Success"], fontsize=8)
        acc = accuracy_score(y_true, yp)
        ax.set_title(f"{name}\nAcc={acc:.3f}", fontweight="bold", fontsize=9, color=col)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("📦 Quik SKU Success Predictor")
st.caption("Pre-listing model — No Availability / ROS signals | LR · RF · GBM ensemble")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    avail_threshold = st.slider(
        "Min Availability % (0 = no filter)", 0, 100, 0, 5,
        help="Filter out SKUs below this availability before scoring. Training used 70%."
    )
    avail_filter = avail_threshold > 0

    st.markdown("**Decision threshold** *(applies to all models)*")
    global_threshold = st.slider(
        "Global threshold", 0.30, 0.70, 0.50, 0.05,
        help="Default probability cutoff for Success/Fail — overridable per model below."
    )

    with st.expander("🔧 Per-model threshold overrides"):
        st.caption("Leave at 0.00 to use the global threshold above.")
        lr_thresh_raw  = st.slider("Logistic Regression", 0.00, 0.70, 0.00, 0.05, key="lr_t")
        rf_thresh_raw  = st.slider("Random Forest",        0.00, 0.70, 0.00, 0.05, key="rf_t")
        gbm_thresh_raw = st.slider("Gradient Boosting",    0.00, 0.70, 0.00, 0.05, key="gbm_t")

    model_thresholds = {
        "Logistic Regression": lr_thresh_raw  if lr_thresh_raw  > 0 else global_threshold,
        "Random Forest":       rf_thresh_raw  if rf_thresh_raw  > 0 else global_threshold,
        "Gradient Boosting":   gbm_thresh_raw if gbm_thresh_raw > 0 else global_threshold,
    }
    # Ensemble uses the mean of the three model thresholds
    threshold = global_threshold  # kept for legacy use in plots

    show_features = st.checkbox("Show engineered features", value=False)
    st.divider()
    st.header("📁 Upload Sheets")
    uploaded_files = st.file_uploader(
        "Upload disposition Excel files",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="Real-time sheets (score columns only) or full backtest sheets with Success column",
    )
    st.divider()
    st.caption("Trained on: Jul25 · Sep25 · Oct25 · Nov25 · Dec25  \nFilter: Avail > 70%  |  30 pre-listing features")

# Train
models, train_info, train_error = train_models()
if train_error:
    st.error(f"⚠️ Could not train models: {train_error}")
    st.stop()

# Training info banner
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training SKUs", f"{train_info['n_skus']:,}")
c2.metric("Training Months", train_info["n_months"])
c3.metric("Historical Success Rate", f"{train_info['success_rate']:.1%}")
c4.metric("Models Ready", "3 ✅")
if train_info["missing_months"]:
    st.warning(f"Missing backtest files: {train_info['missing_months']}")

st.divider()

if not uploaded_files:
    st.info("👈 Upload one or more disposition Excel files from the sidebar to get predictions.")
    with st.expander("ℹ️ Supported file formats"):
        st.markdown("""
        **Real-time disposition sheet** (standard use):
        - Contains score columns: `Vel Score`, `CQ Score`, `Benchmark`, etc.
        - No `Success` column needed — purely predictive
        - App will show predictions + confidence for each SKU

        **Backtest sheet** (evaluation mode):
        - Same as above but with `Success` column (0/1 labels)
        - App additionally shows accuracy, confusion matrix, vs rule-based comparison

        Both formats are auto-detected. Headers on row 5 (index 4) or 6 work.
        `Refrence Data` sheet is used for Subcat P50 lookups if present.
        """)
    st.stop()

# ── Process files ─────────────────────────────────────────────────────────────
all_results = []
tab_names = [f.name for f in uploaded_files] + (["📊 Combined"] if len(uploaded_files) > 1 else [])
tabs = st.tabs(tab_names)

for tab, uploaded_file in zip(tabs[:len(uploaded_files)], uploaded_files):
    with tab:
        with st.spinner(f"Processing {uploaded_file.name}…"):
            file_bytes = uploaded_file.read()
            df, has_labels, has_rule, missing_scores, err = load_new_sheet(file_bytes)

        if err:
            st.error(f"Failed to load file: {err}")
            continue

        # Mode banner
        if has_labels:
            st.success("📋 **Backtest mode** — `Success` column detected. Showing predictions + accuracy analysis.")
        else:
            st.info("🚀 **Real-time mode** — No ground truth available. Showing predictions only.")

        if missing_scores:
            st.warning(f"⚠️ These score columns were not found (filled with 0): `{'`, `'.join(missing_scores)}`")

        # Availability filter
        if avail_filter:
            before = len(df)
            df = df[df["Availability"] > avail_threshold / 100].reset_index(drop=True)
            st.caption(f"Availability filter (>{avail_threshold}%): {before} → {len(df)} SKUs")

        if len(df) == 0:
            st.warning("No rows remaining after filtering.")
            continue

        # Run predictions
        probas, preds = predict(models, df, model_thresholds, global_threshold)
        out_df, id_cols = build_output(df, probas, preds, global_threshold, model_thresholds, has_labels, has_rule)

        # ── Summary metrics ────────────────────────────────────────────────────
        ens_p = probas["Ensemble"]
        n_success = (ens_p >= threshold).sum()
        n_fail    = len(df) - n_success

        cols = st.columns(5)
        cols[0].metric("Total SKUs",        len(df))
        cols[1].metric("Predicted Success", n_success, f"{n_success/len(df):.0%}")
        cols[2].metric("Predicted Fail",    n_fail,    f"{n_fail/len(df):.0%}")
        cols[3].metric("Avg Success Prob",  f"{ens_p.mean():.2f}")
        cols[4].metric("High Confidence",
                       int(((ens_p > 0.7) | (ens_p < 0.3)).sum()),
                       f"{((ens_p > 0.7) | (ens_p < 0.3)).mean():.0%}")

        st.divider()

        # ── Prediction table ───────────────────────────────────────────────────
        st.subheader("🔮 Predictions")

        pred_display_cols = id_cols.copy()
        # Core score columns for context
        context_cols = [c for c in ["CQ\n SCORE", "Vel\n Score", "Benchmark", "Availability"]
                        if c in out_df.columns]
        pred_display_cols += context_cols
        # Model probabilities and predictions
        for name in MODEL_META:
            pred_display_cols += [f"{name} Prob", f"{MODEL_META[name]['short']} Pred"]
        pred_display_cols += ["Ensemble Prob", "Final Prediction", "Confidence", "Model Votes (of 3)"]
        if has_rule:
            pred_display_cols.append("Rule Pred")

        display_df = out_df[[c for c in pred_display_cols if c in out_df.columns]].copy()

        def style_prob(val):
            try:
                v = float(val)
                if   v >= 0.70: return "background-color:#C8E6C9;color:#1B5E20;font-weight:bold"
                elif v >= 0.55: return "background-color:#DCEDC8"
                elif v <= 0.30: return "background-color:#FFCDD2;color:#B71C1C;font-weight:bold"
                elif v <= 0.45: return "background-color:#FFE0B2"
                return "background-color:#F5F5F5"
            except:
                return ""

        def style_label(val):
            if "✅" in str(val): return "color:#2E7D32;font-weight:bold"
            if "❌" in str(val): return "color:#C62828;font-weight:bold"
            return ""

        def style_conf(val):
            if val == "High":   return "color:#1565C0;font-weight:bold"
            if val == "Medium": return "color:#E65100"
            return "color:#888888"

        prob_cols    = [c for c in display_df.columns if "Prob" in c]
        pred_cols    = [c for c in display_df.columns if "Pred" in c or "Prediction" in c]
        conf_cols    = [c for c in display_df.columns if c == "Confidence"]

        styled = display_df.style\
            .applymap(style_prob,  subset=prob_cols)\
            .applymap(style_label, subset=pred_cols)\
            .applymap(style_conf,  subset=conf_cols)

        st.dataframe(styled, use_container_width=True, height=460)

        # ── Probability distribution chart ─────────────────────────────────────
        st.subheader("📈 Probability Distribution")
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=False)
        fig.suptitle("Success Probability per Model", fontsize=11, fontweight="bold")

        for ax, (mname, meta) in zip(axes, MODEL_META.items()):
            p = probas[mname]
            if has_labels:
                y_true = df["Success"].dropna().isin([1, 1.0]).astype(int).reindex(df.index, fill_value=0)
                ax.hist(p[y_true == 0], bins=20, color="#C62828", alpha=0.65,
                        edgecolor="white", label="Actual Fail", density=True)
                ax.hist(p[y_true == 1], bins=20, color="#1565C0", alpha=0.65,
                        edgecolor="white", label="Actual Success", density=True)
                ax.legend(fontsize=7)
            else:
                ax.hist(p, bins=20, color=meta["color"], alpha=0.80, edgecolor="white")

            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5)
            n_s = (p >= threshold).sum()
            ax.set_title(f"{mname}", fontsize=9, fontweight="bold", color=meta["color"])
            ax.set_xlabel("P(Success)", fontsize=8)
            ax.set_ylabel("Count" if not has_labels else "Density", fontsize=8)
            ax.text(0.97, 0.96, f"{n_s}/{len(p)} predicted success",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=meta["color"])
            ax.grid(alpha=0.2); ax.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Score feature breakdown (top signals) ─────────────────────────────
        st.subheader("🔑 Key Signal Breakdown")
        signal_cols = {
            "CQ\n SCORE": "CQ Score",
            "Vel\n Score": "Vel Score",
            "Benchmark": "Benchmark",
            "above_floor": "Above Vel Floor",
            "Launch\n Score": "Launch Score",
            "Sat\n Score": "Sat Score",
            "NMI\n Unit%": "NMI Unit%",
        }
        avail_signals = {v: k for k, v in signal_cols.items() if k in df.columns}

        if avail_signals:
            fig2, axes2 = plt.subplots(1, min(4, len(avail_signals)),
                                       figsize=(3.5 * min(4, len(avail_signals)), 3.2))
            if len(avail_signals) == 1:
                axes2 = [axes2]
            fig2.suptitle("Score Distribution — Predicted Success vs Fail",
                          fontsize=10, fontweight="bold")

            for ax, (label, col) in zip(axes2, list(avail_signals.items())[:4]):
                vals = df[col].fillna(0)
                success_mask = probas["Ensemble"] >= threshold
                ax.hist(vals[~success_mask], bins=15, color="#C62828", alpha=0.6,
                        edgecolor="white", density=True, label="Pred Fail")
                ax.hist(vals[success_mask],  bins=15, color="#1565C0", alpha=0.6,
                        edgecolor="white", density=True, label="Pred Success")
                ax.set_title(label, fontsize=9, fontweight="bold")
                ax.set_xlabel("Value", fontsize=8)
                ax.legend(fontsize=7)
                ax.grid(alpha=0.2); ax.spines[["top","right"]].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Accuracy analysis (backtest only) ──────────────────────────────────
        if has_labels:
            st.divider()
            render_accuracy_analysis(df, probas, preds, threshold, has_rule)

        # ── Optional: engineered features ─────────────────────────────────────
        if show_features:
            with st.expander("🔬 Engineered features"):
                feat_df = pd.DataFrame(
                    df[BASE_FEATURES].fillna(0).values,
                    columns=[FEAT_LABELS[f] for f in BASE_FEATURES]
                )
                st.dataframe(feat_df.head(100), use_container_width=True)

        # ── Download ───────────────────────────────────────────────────────────
        st.divider()
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # Clean predictions sheet
            clean_cols = id_cols + [
                "Ensemble Prob", "Final Prediction", "Confidence", "Model Votes (of 3)",
                "Logistic Regression Prob", "LR Pred",
                "Random Forest Prob", "RF Pred",
                "Gradient Boosting Prob", "GBM Pred",
            ] + (["Rule Pred"] if has_rule else []) + (["Success"] if has_labels else [])
            out_df[[c for c in clean_cols if c in out_df.columns]].to_excel(
                writer, sheet_name="Predictions", index=False)
            out_df.to_excel(writer, sheet_name="Full Output", index=False)
        buf.seek(0)
        dl_name = uploaded_file.name.replace(".xlsx","").replace(".xls","") + "_predictions.xlsx"
        st.download_button(
            f"⬇️ Download predictions — {uploaded_file.name}",
            data=buf, file_name=dl_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        out_df["_source"] = uploaded_file.name
        all_results.append(out_df)

# ── Combined tab ──────────────────────────────────────────────────────────────
if len(uploaded_files) > 1 and all_results:
    with tabs[-1]:
        combined = pd.concat(all_results, ignore_index=True)
        st.subheader(f"Combined: {len(combined):,} SKUs across {len(all_results)} file(s)")

        c1, c2, c3 = st.columns(3)
        n_s = (combined["Ensemble Prob"] >= threshold).sum()
        c1.metric("Total SKUs",        len(combined))
        c2.metric("Predicted Success", n_s, f"{n_s/len(combined):.0%}")
        c3.metric("Predicted Fail",    len(combined)-n_s)

        st.subheader("Per-file Summary")
        breakdown = combined.groupby("_source").agg(
            SKUs=("Ensemble Prob", "count"),
            Avg_Prob=("Ensemble Prob", "mean"),
            Predicted_Success=("Ensemble Prob", lambda x: (x >= threshold).sum()),
        ).reset_index()
        breakdown.rename(columns={"_source": "File"}, inplace=True)
        breakdown["Success Rate"] = (breakdown["Predicted_Success"] / breakdown["SKUs"]).map("{:.1%}".format)
        breakdown["Avg_Prob"]     = breakdown["Avg_Prob"].map("{:.3f}".format)
        st.dataframe(breakdown, use_container_width=True)

        buf2 = io.BytesIO()
        with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
            combined.drop(columns=["_source"]).to_excel(writer, sheet_name="All Predictions", index=False)
            breakdown.to_excel(writer, sheet_name="Summary", index=False)
        buf2.seek(0)
        st.download_button(
            "⬇️ Download combined predictions",
            data=buf2, file_name="quik_sku_combined_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
