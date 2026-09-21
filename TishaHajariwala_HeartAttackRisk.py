"""
Heart Attack Risk Prediction System
=====================================
Single-file Python application combining:
  - Data loading & quality checks
  - Exploratory Data Analysis (saves 12 charts to eda_output/)
  - Machine learning model training (Random Forest)
  - Flask web application (frontend + backend + REST API)

Usage:
    python main.py              → run web app  (default)
    python main.py --eda        → run EDA + quality report only (no server)

Routes (web mode):
    http://localhost:5000/            Patient prediction form
    http://localhost:5000/eda         Live EDA dashboard
    http://localhost:5000/model_info  Model performance metrics
    http://localhost:5000/dataset_info Dataset stats & preview
    http://localhost:5000/api/predict  JSON REST endpoint (POST)
"""

import os
import sys
import io
import base64
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from flask import Flask, render_template_string, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH   = "heart_attack_risk_dataset_20k.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
EDA_OUTPUT_DIR = "eda_output"

KEY_NUM_COLS = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate",
    "total_cholesterol", "ldl_cholesterol", "hdl_cholesterol",
    "triglycerides", "fasting_glucose", "hba1c", "crp_mg_l",
    "stress_level", "risk_probability",
]

FEATURE_COLS = [
    "age", "gender_enc", "bmi", "smoking", "alcohol_consumption",
    "activity_enc", "diet_enc", "stress_level",
    "systolic_bp", "diastolic_bp", "heart_rate",
    "total_cholesterol", "ldl_cholesterol", "hdl_cholesterol",
    "triglycerides", "fasting_glucose", "hba1c", "crp_mg_l",
    "diabetes", "hypertension", "family_history_heart",
    "prev_heart_disease", "stroke_history", "kidney_disease",
    "ecg_abnormal", "left_ventricular_hypertrophy",
    "chest_pain", "shortness_of_breath", "fatigue", "dizziness",
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA QUALITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def run_data_quality(df):
    """Print a full data quality report to stdout."""
    print("=" * 70)
    print("HEART ATTACK RISK DATASET — DATA QUALITY & EDA REPORT")
    print("=" * 70)
    print(f"\nDataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns       : {list(df.columns)}\n")

    print("-" * 70)
    print("SECTION 1 - DATA QUALITY CHECKS")
    print("-" * 70)

    # 1a. Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    if missing_df.empty:
        print("\n[OK] No missing values found in the dataset.")
    else:
        print(f"\n[WARN] Columns with missing values:\n{missing_df}")

    # 1b. Duplicate rows
    n_dupes = df.duplicated().sum()
    print(f"\nDuplicate rows : {n_dupes}")

    # 1c. Data types
    print("\nData types:")
    print(df.dtypes.to_string())

    # 1d. Descriptive statistics
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nDescriptive statistics (numerical columns):")
    print(df[num_cols].describe().round(2).to_string())

    # 1e. Categorical value counts
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print("\nCategorical columns - value counts:")
    for col in cat_cols:
        print(f"\n  {col}: {df[col].value_counts().to_dict()}")

    # 1f. Outlier detection (IQR method)
    print("\nOutlier detection (IQR method):")
    outlier_summary = {}
    for col in KEY_NUM_COLS:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = n_out
        if n_out > 0:
            print(f"  {col:<28}: {n_out:>5} outliers  (range {lower:.2f} - {upper:.2f})")
    if all(v == 0 for v in outlier_summary.values()):
        print("  [OK] No outliers detected in key numerical columns.")

    # 1g. Logical consistency checks
    print("\nConsistency checks:")
    ldl_hdl_issue = (df["ldl_cholesterol"] + df["hdl_cholesterol"] > df["total_cholesterol"]).sum()
    print(f"  LDL + HDL > Total Cholesterol (possible inconsistency) : {ldl_hdl_issue} rows")
    bp_issue = (df["diastolic_bp"] >= df["systolic_bp"]).sum()
    print(f"  Diastolic BP >= Systolic BP (anomaly)                  : {bp_issue} rows")
    bmi_extreme = ((df["bmi"] < 10) | (df["bmi"] > 70)).sum()
    print(f"  BMI outside realistic range 10-70                      : {bmi_extreme} rows")
    age_extreme = ((df["age"] < 0) | (df["age"] > 120)).sum()
    print(f"  Age outside range 0-120                                : {age_extreme} rows")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EDA VISUALISATIONS (save to disk)
# ─────────────────────────────────────────────────────────────────────────────
def run_eda_to_disk(df):
    """Generate 12 EDA chart PNGs and save them to eda_output/."""
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")

    print("\n" + "-" * 70)
    print("SECTION 2 - EXPLORATORY DATA ANALYSIS")
    print("-" * 70)

    # Chart 1 — Target distribution
    print("\n[EDA]  1. Target variable distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    risk_counts = df["heart_attack_risk"].value_counts()
    axes[0].pie(risk_counts, labels=["No Risk", "Risk"], autopct="%1.1f%%",
                colors=["#4CAF50", "#F44336"], startangle=90)
    axes[0].set_title("Heart Attack Risk Distribution")
    sns.countplot(x="risk_category", data=df, ax=axes[1], palette="Set2",
                  order=["Low", "Moderate", "High"])
    axes[1].set_title("Risk Category Counts")
    axes[1].set_xlabel("Risk Category")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/01_target_distribution.png", dpi=120)
    plt.close()

    # Chart 2 — Age distribution by risk
    print("[EDA]  2. Age distribution by risk")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["age"].hist(bins=30, ax=axes[0], color="#5C85D6", edgecolor="white")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    sns.boxplot(x="heart_attack_risk", y="age", data=df, ax=axes[1],
                palette=["#4CAF50", "#F44336"])
    axes[1].set_xticklabels(["No Risk", "Risk"])
    axes[1].set_title("Age vs Heart Attack Risk")
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/02_age_distribution.png", dpi=120)
    plt.close()

    # Chart 3 — Gender breakdown
    print("[EDA]  3. Gender breakdown")
    fig, ax = plt.subplots(figsize=(7, 4))
    gender_risk = df.groupby(["gender", "heart_attack_risk"]).size().unstack(fill_value=0)
    gender_risk.columns = ["No Risk", "Risk"]
    gender_risk.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"], edgecolor="white")
    ax.set_title("Gender vs Heart Attack Risk")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/03_gender_risk.png", dpi=120)
    plt.close()

    # Chart 4 — BMI distribution
    print("[EDA]  4. BMI distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["bmi"].hist(bins=30, ax=axes[0], color="#FF9800", edgecolor="white")
    axes[0].set_title("BMI Distribution")
    axes[0].set_xlabel("BMI")
    sns.boxplot(x="heart_attack_risk", y="bmi", data=df, ax=axes[1],
                palette=["#4CAF50", "#F44336"])
    axes[1].set_xticklabels(["No Risk", "Risk"])
    axes[1].set_title("BMI vs Heart Attack Risk")
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/04_bmi_distribution.png", dpi=120)
    plt.close()

    # Chart 5 — Blood pressure
    print("[EDA]  5. Blood pressure analysis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df, x="systolic_bp", hue="heart_attack_risk", bins=30, ax=axes[0],
                 palette=["#4CAF50", "#F44336"], alpha=0.7)
    axes[0].set_title("Systolic BP Distribution by Risk")
    sns.histplot(df, x="diastolic_bp", hue="heart_attack_risk", bins=30, ax=axes[1],
                 palette=["#4CAF50", "#F44336"], alpha=0.7)
    axes[1].set_title("Diastolic BP Distribution by Risk")
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/05_blood_pressure.png", dpi=120)
    plt.close()

    # Chart 6 — Cholesterol
    print("[EDA]  6. Cholesterol analysis")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color in zip(axes,
                               ["total_cholesterol", "ldl_cholesterol", "hdl_cholesterol"],
                               ["#9C27B0", "#E91E63", "#00BCD4"]):
        sns.histplot(df[col], bins=30, ax=ax, color=color, kde=True)
        ax.set_title(col.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/06_cholesterol.png", dpi=120)
    plt.close()

    # Chart 7 — Lifestyle factors
    print("[EDA]  7. Lifestyle factors")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["smoking", "alcohol_consumption", "physical_activity"]):
        if df[col].dtype == "object":
            sns.countplot(x=col, data=df, hue="heart_attack_risk", ax=ax,
                          palette=["#4CAF50", "#F44336"],
                          order=df[col].value_counts().index)
        else:
            risk_pct = df.groupby(col)["heart_attack_risk"].mean() * 100
            risk_pct.plot(kind="bar", ax=ax, color="#5C85D6", edgecolor="white")
            ax.set_ylabel("Risk %")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/07_lifestyle.png", dpi=120)
    plt.close()

    # Chart 8 — Correlation heatmap
    print("[EDA]  8. Correlation heatmap")
    fig, ax = plt.subplots(figsize=(16, 12))
    corr = df[KEY_NUM_COLS + ["heart_attack_risk"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/08_correlation_heatmap.png", dpi=120)
    plt.close()

    # Chart 9 — Risk probability
    print("[EDA]  9. Risk probability distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["risk_probability"].hist(bins=40, ax=axes[0], color="#607D8B", edgecolor="white")
    axes[0].set_title("Risk Probability Distribution")
    axes[0].set_xlabel("Risk Probability")
    sns.boxplot(x="risk_category", y="risk_probability", data=df, ax=axes[1],
                order=["Low", "Moderate", "High"], palette="Set2")
    axes[1].set_title("Risk Probability by Category")
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/09_risk_probability.png", dpi=120)
    plt.close()

    # Chart 10 — Comorbidities
    print("[EDA] 10. Comorbidities vs risk")
    binary_cols = [c for c in [
        "diabetes", "hypertension", "family_history_heart", "prev_heart_disease",
        "stroke_history", "kidney_disease", "ecg_abnormal", "left_ventricular_hypertrophy",
        "chest_pain", "shortness_of_breath", "fatigue", "dizziness",
    ] if c in df.columns]
    risk_rates = {
        col: {
            "With Condition":    df[df[col] == 1]["heart_attack_risk"].mean() * 100,
            "Without Condition": df[df[col] == 0]["heart_attack_risk"].mean() * 100,
        }
        for col in binary_cols
    }
    fig, ax = plt.subplots(figsize=(14, 6))
    pd.DataFrame(risk_rates).T.plot(kind="bar", ax=ax,
                                     color=["#F44336", "#4CAF50"], edgecolor="white")
    ax.set_title("Heart Attack Risk % - With vs Without Comorbidity/Symptom")
    ax.set_ylabel("Risk %")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/10_comorbidities.png", dpi=120)
    plt.close()

    # Chart 11 — Stress level vs risk
    print("[EDA] 11. Stress level vs risk")
    fig, ax = plt.subplots(figsize=(10, 4))
    stress_risk = df.groupby("stress_level")["heart_attack_risk"].mean() * 100
    stress_risk.plot(kind="bar", ax=ax, color="#FF5722", edgecolor="white")
    ax.set_title("Heart Attack Risk % by Stress Level")
    ax.set_xlabel("Stress Level (1-10)")
    ax.set_ylabel("Risk %")
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/11_stress_vs_risk.png", dpi=120)
    plt.close()

    # Chart 12 — Diet quality vs risk
    print("[EDA] 12. Diet quality vs risk")
    diet_order = [o for o in ["Poor", "Average", "Good", "Excellent"]
                  if o in df["diet_quality"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x="diet_quality", y="heart_attack_risk", data=df, ax=ax,
                order=diet_order, palette="RdYlGn", ci=None)
    ax.set_title("Heart Attack Risk Rate by Diet Quality")
    ax.set_ylabel("Risk Rate")
    ax.set_xlabel("Diet Quality")
    plt.tight_layout()
    plt.savefig(f"{EDA_OUTPUT_DIR}/12_diet_quality.png", dpi=120)
    plt.close()

    print(f"\n[OK] All 12 EDA charts saved to '{EDA_OUTPUT_DIR}/'")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA PREPROCESSING & MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df):
    """Encode categoricals and return feature matrix X and target y."""
    df = df.copy()
    df["gender_enc"]   = LabelEncoder().fit_transform(df["gender"])
    df["activity_enc"] = LabelEncoder().fit_transform(df["physical_activity"])
    df["diet_enc"]     = LabelEncoder().fit_transform(df["diet_quality"])
    X = df[FEATURE_COLS]
    y = df["heart_attack_risk"]
    return X, y


def train_model(X, y):
    """Train Random Forest, return (model, scaler, metrics_dict)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_split=5, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "roc_auc":  round(roc_auc_score(y_test, y_proba) * 100, 2),
        "report":   classification_report(y_test, y_pred, output_dict=True),
        "cm":       confusion_matrix(y_test, y_pred).tolist(),
        "fpr":      fpr.tolist(),
        "tpr":      tpr.tolist(),
    }

    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    return model, scaler, metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — IN-MEMORY EDA CHARTS FOR THE WEB DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def build_web_eda_charts(df):
    """Return a dict of base64-encoded PNG strings for the web EDA page."""
    charts = {}
    sns.set_theme(style="whitegrid", palette="Set2")

    # Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    vals = df["heart_attack_risk"].value_counts()
    axes[0].pie(vals, labels=["No Risk (0)", "Risk (1)"], autopct="%1.1f%%",
                colors=["#4CAF50", "#F44336"], startangle=90)
    axes[0].set_title("Heart Attack Risk Distribution")
    sns.countplot(x="risk_category", data=df, ax=axes[1], palette="Set2",
                  order=["Low", "Moderate", "High"])
    axes[1].set_title("Risk Category Counts")
    plt.tight_layout()
    charts["target"] = _fig_to_b64(fig)

    # Age & BMI
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df, x="age", hue="heart_attack_risk", bins=30, ax=axes[0],
                 palette=["#4CAF50", "#F44336"], alpha=0.7)
    axes[0].set_title("Age Distribution by Risk")
    sns.boxplot(x="heart_attack_risk", y="bmi", data=df, ax=axes[1],
                palette=["#4CAF50", "#F44336"])
    axes[1].set_xticklabels(["No Risk", "Risk"])
    axes[1].set_title("BMI vs Heart Attack Risk")
    plt.tight_layout()
    charts["age_bmi"] = _fig_to_b64(fig)

    # Correlation heatmap
    heatmap_cols = KEY_NUM_COLS + ["heart_attack_risk"]
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df[heatmap_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    charts["corr"] = _fig_to_b64(fig)

    # Gender vs Risk
    fig, ax = plt.subplots(figsize=(7, 4))
    g = df.groupby(["gender", "heart_attack_risk"]).size().unstack(fill_value=0)
    g.columns = ["No Risk", "Risk"]
    g.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"], edgecolor="white")
    ax.set_title("Gender vs Heart Attack Risk")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    charts["gender"] = _fig_to_b64(fig)

    # BP scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4CAF50" if r == 0 else "#F44336" for r in df["heart_attack_risk"]]
    ax.scatter(df["systolic_bp"], df["diastolic_bp"], c=colors, alpha=0.3, s=10)
    ax.set_xlabel("Systolic BP")
    ax.set_ylabel("Diastolic BP")
    ax.set_title("Blood Pressure Scatter (Green=No Risk, Red=Risk)")
    plt.tight_layout()
    charts["bp"] = _fig_to_b64(fig)

    return charts


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FLASK WEB APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "heart_risk_2024"

# Globals populated at startup
DF = MODEL = SCALER = TRAIN_METRICS = EDA_CHARTS = None

BASE_CSS = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body { background:#f0f4f8; font-family:'Segoe UI',sans-serif; }
  .navbar { background:linear-gradient(135deg,#c0392b,#922b21); }
  .card   { border:none; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08); }
  .card-header { border-radius:12px 12px 0 0 !important; }
  .risk-high { background:#F44336; color:#fff; border-radius:8px; padding:16px; }
  .risk-mod  { background:#FF9800; color:#fff; border-radius:8px; padding:16px; }
  .risk-low  { background:#4CAF50; color:#fff; border-radius:8px; padding:16px; }
  .metric-box { text-align:center; padding:20px; border-radius:10px; background:#fff;
                box-shadow:0 2px 8px rgba(0,0,0,.07); }
  .metric-val { font-size:2rem; font-weight:700; }
  .nav-link   { color:#fff !important; }
  img.eda-img { width:100%; border-radius:8px; margin-bottom:12px; }
</style>
"""

NAV = """
<nav class="navbar navbar-expand-lg mb-4">
  <div class="container-fluid">
    <a class="navbar-brand text-white fw-bold fs-5" href="/">&#10084; Heart Attack Risk Predictor</a>
    <div class="navbar-nav ms-auto">
      <a class="nav-link" href="/">Predict</a>
      <a class="nav-link" href="/eda">EDA Dashboard</a>
      <a class="nav-link" href="/model_info">Model Info</a>
      <a class="nav-link" href="/dataset_info">Dataset Info</a>
    </div>
  </div>
</nav>
"""

PREDICT_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>Heart Attack Risk Predictor</title>""" + BASE_CSS + """</head>
<body>""" + NAV + """
<div class="container pb-5">
  <div class="row justify-content-center">
    <div class="col-lg-9">
      <div class="card">
        <div class="card-header bg-danger text-white py-3">
          <h4 class="mb-0">&#129658; Patient Risk Assessment</h4>
          <small>Enter patient details to predict heart attack risk</small>
        </div>
        <div class="card-body p-4">
          {% if result %}
          <div class="{{ result.css }} mb-4">
            <h4>{{ result.emoji }} Prediction: {{ result.label }}</h4>
            <p class="mb-1">Risk Probability: <strong>{{ result.prob }}%</strong></p>
            <p class="mb-0">{{ result.advice }}</p>
          </div>
          {% endif %}
          <form method="POST" action="/predict">
            <div class="row g-3">
              <div class="col-12"><h6 class="text-muted border-bottom pb-1">Demographics</h6></div>
              <div class="col-md-3">
                <label class="form-label">Age</label>
                <input type="number" class="form-control" name="age" min="18" max="100" value="{{ vals.age or 45 }}" required>
              </div>
              <div class="col-md-3">
                <label class="form-label">Gender</label>
                <select class="form-select" name="gender">
                  <option value="Male"   {% if vals.gender == 'Male'   %}selected{% endif %}>Male</option>
                  <option value="Female" {% if vals.gender == 'Female' %}selected{% endif %}>Female</option>
                </select>
              </div>
              <div class="col-md-3">
                <label class="form-label">BMI</label>
                <input type="number" step="0.1" class="form-control" name="bmi" min="10" max="70" value="{{ vals.bmi or 25.0 }}" required>
              </div>
              <div class="col-md-3">
                <label class="form-label">Stress Level (1-10)</label>
                <input type="number" class="form-control" name="stress_level" min="1" max="10" value="{{ vals.stress_level or 5 }}" required>
              </div>
              <div class="col-12"><h6 class="text-muted border-bottom pb-1">Vitals</h6></div>
              <div class="col-md-4">
                <label class="form-label">Systolic BP</label>
                <input type="number" class="form-control" name="systolic_bp" value="{{ vals.systolic_bp or 120 }}" required>
              </div>
              <div class="col-md-4">
                <label class="form-label">Diastolic BP</label>
                <input type="number" class="form-control" name="diastolic_bp" value="{{ vals.diastolic_bp or 80 }}" required>
              </div>
              <div class="col-md-4">
                <label class="form-label">Heart Rate</label>
                <input type="number" class="form-control" name="heart_rate" value="{{ vals.heart_rate or 72 }}" required>
              </div>
              <div class="col-12"><h6 class="text-muted border-bottom pb-1">Lab Results</h6></div>
              <div class="col-md-3">
                <label class="form-label">Total Cholesterol (mg/dL)</label>
                <input type="number" step="0.1" class="form-control" name="total_cholesterol" value="{{ vals.total_cholesterol or 200.0 }}" required>
              </div>
              <div class="col-md-3">
                <label class="form-label">LDL Cholesterol</label>
                <input type="number" step="0.1" class="form-control" name="ldl_cholesterol" value="{{ vals.ldl_cholesterol or 120.0 }}" required>
              </div>
              <div class="col-md-3">
                <label class="form-label">HDL Cholesterol</label>
                <input type="number" step="0.1" class="form-control" name="hdl_cholesterol" value="{{ vals.hdl_cholesterol or 50.0 }}" required>
              </div>
              <div class="col-md-3">
                <label class="form-label">Triglycerides</label>
                <input type="number" step="0.1" class="form-control" name="triglycerides" value="{{ vals.triglycerides or 150.0 }}" required>
              </div>
              <div class="col-md-4">
                <label class="form-label">Fasting Glucose</label>
                <input type="number" step="0.1" class="form-control" name="fasting_glucose" value="{{ vals.fasting_glucose or 100.0 }}" required>
              </div>
              <div class="col-md-4">
                <label class="form-label">HbA1c (%)</label>
                <input type="number" step="0.1" class="form-control" name="hba1c" value="{{ vals.hba1c or 5.5 }}" required>
              </div>
              <div class="col-md-4">
                <label class="form-label">CRP (mg/L)</label>
                <input type="number" step="0.01" class="form-control" name="crp_mg_l" value="{{ vals.crp_mg_l or 1.0 }}" required>
              </div>
              <div class="col-12"><h6 class="text-muted border-bottom pb-1">Lifestyle</h6></div>
              <div class="col-md-3">
                <label class="form-label">Physical Activity</label>
                <select class="form-select" name="physical_activity">
                  <option value="Low">Low</option>
                  <option value="Moderate" selected>Moderate</option>
                  <option value="High">High</option>
                </select>
              </div>
              <div class="col-md-3">
                <label class="form-label">Diet Quality</label>
                <select class="form-select" name="diet_quality">
                  <option value="Poor">Poor</option>
                  <option value="Average" selected>Average</option>
                  <option value="Good">Good</option>
                </select>
              </div>
              <div class="col-md-3">
                <label class="form-label">Smoking</label>
                <select class="form-select" name="smoking">
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>
              <div class="col-md-3">
                <label class="form-label">Alcohol Consumption</label>
                <select class="form-select" name="alcohol_consumption">
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>
              <div class="col-12"><h6 class="text-muted border-bottom pb-1">Medical History &amp; Symptoms</h6></div>
              {% for name, label in [
                ('diabetes','Diabetes'),('hypertension','Hypertension'),
                ('family_history_heart','Family History (Heart)'),
                ('prev_heart_disease','Previous Heart Disease'),
                ('stroke_history','Stroke History'),
                ('kidney_disease','Kidney Disease'),
                ('ecg_abnormal','ECG Abnormal'),
                ('left_ventricular_hypertrophy','LV Hypertrophy'),
                ('chest_pain','Chest Pain'),
                ('shortness_of_breath','Shortness of Breath'),
                ('fatigue','Fatigue'),('dizziness','Dizziness')] %}
              <div class="col-md-3">
                <label class="form-label">{{ label }}</label>
                <select class="form-select" name="{{ name }}">
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>
              {% endfor %}
              <div class="col-12 mt-3">
                <button type="submit" class="btn btn-danger btn-lg px-5">&#128269; Predict Risk</button>
                <a href="/" class="btn btn-outline-secondary btn-lg ms-2">Reset</a>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  </div>
</div>
</body></html>"""

EDA_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>EDA Dashboard</title>""" + BASE_CSS + """</head>
<body>""" + NAV + """
<div class="container pb-5">
  <h3 class="mb-4">&#128202; Exploratory Data Analysis Dashboard</h3>
  <div class="row g-4">
    <div class="col-md-12"><div class="card"><div class="card-body">
      <h6 class="card-title">Target Distribution</h6>
      <img class="eda-img" src="data:image/png;base64,{{ charts.target }}">
    </div></div></div>
    <div class="col-md-12"><div class="card"><div class="card-body">
      <h6 class="card-title">Age Distribution &amp; BMI vs Risk</h6>
      <img class="eda-img" src="data:image/png;base64,{{ charts.age_bmi }}">
    </div></div></div>
    <div class="col-md-6"><div class="card"><div class="card-body">
      <h6 class="card-title">Gender vs Risk</h6>
      <img class="eda-img" src="data:image/png;base64,{{ charts.gender }}">
    </div></div></div>
    <div class="col-md-6"><div class="card"><div class="card-body">
      <h6 class="card-title">Blood Pressure Scatter</h6>
      <img class="eda-img" src="data:image/png;base64,{{ charts.bp }}">
    </div></div></div>
    <div class="col-md-12"><div class="card"><div class="card-body">
      <h6 class="card-title">Correlation Heatmap</h6>
      <img class="eda-img" src="data:image/png;base64,{{ charts.corr }}">
    </div></div></div>
  </div>
</div></body></html>"""

MODEL_INFO_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>Model Info</title>""" + BASE_CSS + """</head>
<body>""" + NAV + """
<div class="container pb-5">
  <h3 class="mb-4">&#129302; Model Performance</h3>
  <div class="row g-4 mb-4">
    <div class="col-md-4"><div class="metric-box">
      <div class="metric-val text-success">{{ metrics.accuracy }}%</div><div>Accuracy</div>
    </div></div>
    <div class="col-md-4"><div class="metric-box">
      <div class="metric-val text-primary">{{ metrics.roc_auc }}%</div><div>ROC-AUC Score</div>
    </div></div>
    <div class="col-md-4"><div class="metric-box">
      <div class="metric-val text-warning">Random Forest</div><div>Algorithm (200 trees)</div>
    </div></div>
  </div>
  <div class="card mb-4">
    <div class="card-header bg-dark text-white">Classification Report</div>
    <div class="card-body">
      <table class="table table-sm table-bordered">
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
        <tbody>
          {% for cls, row in metrics.report.items() %}
            {% if cls not in ['accuracy','macro avg','weighted avg'] %}
            <tr>
              <td>{{ "No Risk" if cls == "0" else "Risk" }}</td>
              <td>{{ "%.3f"|format(row.precision) }}</td>
              <td>{{ "%.3f"|format(row.recall) }}</td>
              <td>{{ "%.3f"|format(row["f1-score"]) }}</td>
              <td>{{ row.support|int }}</td>
            </tr>
            {% endif %}
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card-header bg-dark text-white">Confusion Matrix</div>
    <div class="card-body">
      <table class="table table-sm table-bordered text-center" style="max-width:320px">
        <thead><tr><th></th><th>Pred: No Risk</th><th>Pred: Risk</th></tr></thead>
        <tbody>
          <tr><td><strong>Actual: No Risk</strong></td>
              <td class="table-success">{{ metrics.cm[0][0] }}</td>
              <td class="table-danger">{{ metrics.cm[0][1] }}</td></tr>
          <tr><td><strong>Actual: Risk</strong></td>
              <td class="table-danger">{{ metrics.cm[1][0] }}</td>
              <td class="table-success">{{ metrics.cm[1][1] }}</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div></body></html>"""

DATASET_INFO_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>Dataset Info</title>""" + BASE_CSS + """</head>
<body>""" + NAV + """
<div class="container pb-5">
  <h3 class="mb-4">&#128193; Dataset Overview</h3>
  <div class="row g-4 mb-4">
    <div class="col-md-3"><div class="metric-box">
      <div class="metric-val text-danger">{{ info.rows }}</div><div>Total Records</div>
    </div></div>
    <div class="col-md-3"><div class="metric-box">
      <div class="metric-val text-primary">{{ info.cols }}</div><div>Features</div>
    </div></div>
    <div class="col-md-3"><div class="metric-box">
      <div class="metric-val text-success">{{ info.risk_pct }}%</div><div>High Risk Patients</div>
    </div></div>
    <div class="col-md-3"><div class="metric-box">
      <div class="metric-val text-warning">{{ info.missing }}</div><div>Missing Values</div>
    </div></div>
  </div>
  <div class="card">
    <div class="card-header bg-dark text-white">Sample Records (first 10)</div>
    <div class="card-body" style="overflow-x:auto">{{ table|safe }}</div>
  </div>
</div></body></html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template_string(PREDICT_TEMPLATE, result=None, vals={})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form
        activity_map = {"Low": 1, "Moderate": 2, "High": 0}
        diet_map     = {"Poor": 2, "Average": 0, "Good": 1}
        gender_map   = {"Male": 1, "Female": 0}

        row = [
            float(form["age"]),
            gender_map.get(form["gender"], 1),
            float(form["bmi"]),
            int(form["smoking"]),
            int(form["alcohol_consumption"]),
            activity_map.get(form["physical_activity"], 2),
            diet_map.get(form["diet_quality"], 0),
            float(form["stress_level"]),
            float(form["systolic_bp"]),
            float(form["diastolic_bp"]),
            float(form["heart_rate"]),
            float(form["total_cholesterol"]),
            float(form["ldl_cholesterol"]),
            float(form["hdl_cholesterol"]),
            float(form["triglycerides"]),
            float(form["fasting_glucose"]),
            float(form["hba1c"]),
            float(form["crp_mg_l"]),
            int(form.get("diabetes", 0)),
            int(form.get("hypertension", 0)),
            int(form.get("family_history_heart", 0)),
            int(form.get("prev_heart_disease", 0)),
            int(form.get("stroke_history", 0)),
            int(form.get("kidney_disease", 0)),
            int(form.get("ecg_abnormal", 0)),
            int(form.get("left_ventricular_hypertrophy", 0)),
            int(form.get("chest_pain", 0)),
            int(form.get("shortness_of_breath", 0)),
            int(form.get("fatigue", 0)),
            int(form.get("dizziness", 0)),
        ]

        X_input = SCALER.transform([row])
        proba   = MODEL.predict_proba(X_input)[0][1] * 100

        if proba >= 60:
            css, emoji, label = "risk-high", "&#128308;", "HIGH RISK"
            advice = "Immediate medical consultation strongly recommended."
        elif proba >= 35:
            css, emoji, label = "risk-mod", "&#128993;", "MODERATE RISK"
            advice = "Consider lifestyle changes and regular monitoring."
        else:
            css, emoji, label = "risk-low", "&#129001;", "LOW RISK"
            advice = "Maintain healthy habits and routine check-ups."

        result = {"label": label, "prob": round(proba, 1),
                  "css": css, "emoji": emoji, "advice": advice}
        return render_template_string(PREDICT_TEMPLATE, result=result, vals=dict(form))

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON REST endpoint for programmatic access."""
    data = request.get_json(force=True)
    activity_map = {"Low": 1, "Moderate": 2, "High": 0}
    diet_map     = {"Poor": 2, "Average": 0, "Good": 1}
    gender_map   = {"Male": 1, "Female": 0}

    row = [
        data.get("age", 45),
        gender_map.get(data.get("gender", "Male"), 1),
        data.get("bmi", 25.0),
        data.get("smoking", 0),
        data.get("alcohol_consumption", 0),
        activity_map.get(data.get("physical_activity", "Moderate"), 2),
        diet_map.get(data.get("diet_quality", "Average"), 0),
        data.get("stress_level", 5),
        data.get("systolic_bp", 120),
        data.get("diastolic_bp", 80),
        data.get("heart_rate", 72),
        data.get("total_cholesterol", 200),
        data.get("ldl_cholesterol", 120),
        data.get("hdl_cholesterol", 50),
        data.get("triglycerides", 150),
        data.get("fasting_glucose", 100),
        data.get("hba1c", 5.5),
        data.get("crp_mg_l", 1.0),
        data.get("diabetes", 0),
        data.get("hypertension", 0),
        data.get("family_history_heart", 0),
        data.get("prev_heart_disease", 0),
        data.get("stroke_history", 0),
        data.get("kidney_disease", 0),
        data.get("ecg_abnormal", 0),
        data.get("left_ventricular_hypertrophy", 0),
        data.get("chest_pain", 0),
        data.get("shortness_of_breath", 0),
        data.get("fatigue", 0),
        data.get("dizziness", 0),
    ]
    X_input = SCALER.transform([row])
    proba   = float(MODEL.predict_proba(X_input)[0][1])
    return jsonify({
        "prediction":       int(proba >= 0.5),
        "risk_probability": round(proba, 4),
        "risk_label":       "High Risk" if proba >= 0.6 else ("Moderate Risk" if proba >= 0.35 else "Low Risk"),
    })


@app.route("/eda")
def eda():
    return render_template_string(EDA_TEMPLATE, charts=EDA_CHARTS)


@app.route("/model_info")
def model_info():
    return render_template_string(MODEL_INFO_TEMPLATE, metrics=TRAIN_METRICS)


@app.route("/dataset_info")
def dataset_info():
    info = {
        "rows":     f"{len(DF):,}",
        "cols":     DF.shape[1],
        "risk_pct": round(DF["heart_attack_risk"].mean() * 100, 1),
        "missing":  int(DF.isnull().sum().sum()),
    }
    table_html = (DF.head(10)
                  .drop(columns=["patient_id"], errors="ignore")
                  .to_html(classes="table table-sm table-striped table-bordered",
                           index=False, border=0))
    return render_template_string(DATASET_INFO_TEMPLATE, info=info, table=table_html)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global DF, MODEL, SCALER, TRAIN_METRICS, EDA_CHARTS

    parser = argparse.ArgumentParser(description="Heart Attack Risk Prediction System")
    parser.add_argument("--eda", action="store_true",
                        help="Run data quality checks + EDA only (no web server)")
    args = parser.parse_args()

    # Always load and quality-check the dataset
    print(f"[INFO] Loading dataset from '{CSV_PATH}' ...")
    DF = pd.read_csv(CSV_PATH)
    run_data_quality(DF)

    if args.eda:
        # EDA-only mode: save all 12 charts to disk and exit
        run_eda_to_disk(DF)
        print("[INFO] EDA complete. Exiting.")
        return

    # Web-app mode: also train model and start server
    run_eda_to_disk(DF)

    print("\n[INFO] Preprocessing and training Random Forest model ...")
    X, y = preprocess(DF)
    MODEL, SCALER, TRAIN_METRICS = train_model(X, y)
    print(f"[INFO] Model ready — Accuracy: {TRAIN_METRICS['accuracy']}%  "
          f"ROC-AUC: {TRAIN_METRICS['roc_auc']}%")

    print("[INFO] Building in-memory EDA charts for web dashboard ...")
    EDA_CHARTS = build_web_eda_charts(DF)

    print("[INFO] Starting Flask server on http://localhost:5000 ...")
    app.run(debug=False, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
