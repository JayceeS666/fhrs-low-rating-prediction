"""
Predicts which food businesses in England are likely to get a low FHRS
hygiene rating, using only things you'd know before an inspection happens
(business type, location, time since last check, area deprivation).

Needs data/FHRS_All_en-GB.csv.gz and data/File_10_IoD2025_...xlsx.
Prints metrics for three models and saves three plots.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

from england_lookup import ENGLAND_REGIONS, IMD_NAME_FIX

FHRS_CSV = "data/FHRS_All_en-GB.csv.gz"
IMD_XLSX = "data/File_10_IoD2025_Local_Authority_District_Summaries.xlsx"
SEED = 42
TEST_SIZE = 0.25
YEARS = 5  # supervisor said older data probably isn't representative


def load_data():
    """Read the raw FHRS export and keep only English FHRS-scheme records.
    (The raw file has no country column, hence the lookup table.)"""
    df = pd.read_csv(FHRS_CSV, dtype=str, low_memory=False)
    england = {la for las in ENGLAND_REGIONS.values() for la in las}
    df = df[(df["SchemeType"] == "FHRS") & df["LocalAuthorityName"].isin(england)].copy()

    # numeric ratings only — drops "Awaiting Inspection", "Exempt" etc.
    df = df[df["RatingValue"].isin(list("012345"))]
    df["RatingValue"] = df["RatingValue"].astype(int)
    df["RatingDate"] = pd.to_datetime(df["RatingDate"], errors="coerce")

    latest = df["RatingDate"].max()
    df = df[df["RatingDate"] >= latest - pd.DateOffset(years=YEARS)]

    # same name + postcode twice = duplicate listing, keep the newer one
    df = df.sort_values("RatingDate").drop_duplicates(["BusinessName", "PostCode"], keep="last")

    df["HighRisk"] = (df["RatingValue"] <= 3).astype(int)
    df["DaysSinceInspection"] = (latest - df["RatingDate"]).dt.days

    # bolt on local-authority deprivation scores
    imd = pd.read_excel(IMD_XLSX, sheet_name="IMD").rename(columns={
        "Local Authority District name (2024)": "LocalAuthorityName",
        "IMD - Average score ": "IMD_DeprivationScore",
        "IMD - Proportion of LSOAs in most deprived 10% nationally ": "IMD_PropMostDeprived10pct",
    })[["LocalAuthorityName", "IMD_DeprivationScore", "IMD_PropMostDeprived10pct"]]
    imd["LocalAuthorityName"] = imd["LocalAuthorityName"].replace(IMD_NAME_FIX)

    df = df.merge(imd, on="LocalAuthorityName", how="left")
    # a couple of port-health authorities have no normal IMD record — drop them
    return df.dropna(subset=["IMD_DeprivationScore"])


def build_features(df):
    """Note: Hygiene / Structural / ConfidenceInManagement are deliberately
    left out. Those three are what the final rating is calculated from, so
    feeding them in would just let the model read the answer off the data."""
    regions = {la: r for r, las in ENGLAND_REGIONS.items() for la in las}
    df["Region"] = df["LocalAuthorityName"].map(regions).fillna("Unknown")

    for col in ["Latitude", "Longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["GeoMissing"] = df["Latitude"].isna().astype(int)
    df["Latitude"] = df["Latitude"].fillna(df["Latitude"].median())
    df["Longitude"] = df["Longitude"].fillna(df["Longitude"].median())

    # lump the rare business types together
    freq = df["BusinessType"].value_counts(normalize=True)
    rare = freq[freq < 0.005].index
    df["BusinessTypeGrp"] = df["BusinessType"].where(~df["BusinessType"].isin(rare), "Other/Rare")

    cols = ["DaysSinceInspection", "Latitude", "Longitude", "GeoMissing",
            "IMD_DeprivationScore", "IMD_PropMostDeprived10pct", "BusinessTypeGrp", "Region"]
    X = pd.get_dummies(df[cols], columns=["BusinessTypeGrp", "Region"])
    return X, df["HighRisk"]


def report(name, y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    print(f"{name} (threshold={threshold:.3f}): AUC={roc_auc_score(y_true, prob):.3f} "
          f"Precision={precision_score(y_true, pred):.3f} "
          f"Recall={recall_score(y_true, pred):.3f} F1={f1_score(y_true, pred):.3f}")
    print(f"  [[TN FP][FN TP]] = {confusion_matrix(y_true, pred).tolist()}")


def train_models(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    # only ~9% are high-risk, so balance the training set — test set is left
    # alone so the metrics still reflect the real class distribution
    X_tr, y_tr = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)

    scaler = StandardScaler().fit(X_tr)
    models = [
        # (name, model, needs_scaling)
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED), True),
        # 300 trees with unlimited depth was far too slow on this much data
        ("Random Forest", RandomForestClassifier(n_estimators=150, max_depth=20,
                                                 min_samples_leaf=5, random_state=SEED, n_jobs=-1), False),
        ("XGBoost", xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                      eval_metric="auc", random_state=SEED, n_jobs=-1,
                                      tree_method="hist"), False),
    ]

    fitted, probs = {}, {}
    for name, model, scale in models:
        model.fit(scaler.transform(X_tr) if scale else X_tr, y_tr)
        probs[name] = model.predict_proba(scaler.transform(X_te) if scale else X_te)[:, 1]
        fitted[name] = model
        report(name, y_te, probs[name])

    # Youden's J = the ROC point with the best TPR/FPR trade-off. Same method
    # Allen et al. (2019) and Oldroyd et al. (2021) used, so results compare.
    fpr, tpr, cuts = roc_curve(y_te, probs["XGBoost"])
    best = cuts[np.argmax(tpr - fpr)]
    report("XGBoost (Youden's J)", y_te, probs["XGBoost"], threshold=best)

    return fitted, probs, X_te, y_te, best


def run_shap(model, X_te, n=5000):
    sample = X_te.sample(n=min(n, len(X_te)), random_state=SEED)
    values = shap.TreeExplainer(model).shap_values(sample)
    importance = pd.Series(np.abs(values).mean(axis=0),
                           index=sample.columns).sort_values(ascending=False)
    print("\nTop 15 features by mean |SHAP value|:")
    print(importance.head(15))
    return values, sample


def make_plots(probs, y_te, best, shap_values, shap_sample):
    plt.figure(figsize=(6.5, 6.5))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_te, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Final Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.close()

    _, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cut, title in [(axes[0], 0.5, "Default threshold (0.5)"),
                           (axes[1], best, f"Youden's J threshold ({best:.3f})")]:
        cm = confusion_matrix(y_te, (probs["XGBoost"] >= cut).astype(int))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([0, 1], ["Pred: Low", "Pred: High"])
        ax.set_yticks([0, 1], ["Actual: Low", "Actual: High"])
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, shap_sample, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_data()
    print(f"  {len(df):,} records, {df['HighRisk'].mean():.2%} high-risk")

    print("Building features...")
    X, y = build_features(df)

    print("Training models...\n")
    models, probs, X_te, y_te, best = train_models(X, y)

    print("\nRunning SHAP on XGBoost...")
    shap_values, shap_sample = run_shap(models["XGBoost"], X_te)

    print("\nMaking plots...")
    make_plots(probs, y_te, best, shap_values, shap_sample)

    print("Done — see roc_curves.png, confusion_matrices.png, shap_summary.png")
