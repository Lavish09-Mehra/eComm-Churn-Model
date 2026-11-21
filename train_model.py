"""Train and package the churn model used by the FastAPI app.

This mirrors the preprocessing and tuned Gradient Boosting model from
``churn_guess.ipynb`` and saves everything the API needs in one bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "eCommerce-dataset.csv"
BUNDLE_PATH = BASE_DIR / "churn_model_bundle.joblib"

TARGET = "Churned"
CATEGORICAL = ["Gender", "Signup_Quarter", "Country"]
DROPPED = ["City"]

# Metrics captured in the notebook's model comparison cell.
NOTEBOOK_RESULTS = {
    "Gradient Boosting": {"accuracy": 0.9190, "precision": 0.937925, "recall": 0.768641, "f1": 0.844887},
    "Random Forest": {"accuracy": 0.9174, "precision": 0.927258, "recall": 0.772822, "f1": 0.843025},
    "XGBoost": {"accuracy": 0.9147, "precision": 0.906817, "recall": 0.783275, "f1": 0.840531},
    "SVM": {"accuracy": 0.8785, "precision": 0.870578, "recall": 0.677352, "f1": 0.761905},
    "Decision Tree": {"accuracy": 0.8440, "precision": 0.722638, "recall": 0.740767, "f1": 0.731590},
    "KNN": {"accuracy": 0.8082, "precision": 0.762404, "recall": 0.481882, "f1": 0.590521},
    "Naive Bayes": {"accuracy": 0.6690, "precision": 0.445518, "recall": 0.626829, "f1": 0.520845},
}

PRETTY_NAMES = {
    "Age": "Age",
    "Membership_Years": "Membership years",
    "Login_Frequency": "Login frequency",
    "Session_Duration_Avg": "Avg. session duration",
    "Pages_Per_Session": "Pages per session",
    "Cart_Abandonment_Rate": "Cart abandonment",
    "Wishlist_Items": "Wishlist items",
    "Total_Purchases": "Total purchases",
    "Average_Order_Value": "Average order value",
    "Days_Since_Last_Purchase": "Days since last purchase",
    "Discount_Usage_Rate": "Discount usage",
    "Returns_Rate": "Returns rate",
    "Email_Open_Rate": "Email open rate",
    "Customer_Service_Calls": "Service calls",
    "Product_Reviews_Written": "Reviews written",
    "Social_Media_Engagement_Score": "Social engagement",
    "Mobile_App_Usage": "Mobile app usage",
    "Payment_Method_Diversity": "Payment methods",
    "Lifetime_Value": "Lifetime value",
    "Credit_Balance": "Credit balance",
}

FIELD_HELP = {
    "Age": {"label": "Age", "group": "Profile", "unit": "years"},
    "Gender": {"label": "Gender", "group": "Profile"},
    "Country": {"label": "Country", "group": "Profile"},
    "City": {"label": "City", "group": "Profile"},
    "Signup_Quarter": {"label": "Signup quarter", "group": "Profile"},
    "Membership_Years": {"label": "Membership years", "group": "Profile", "unit": "years"},
    "Login_Frequency": {"label": "Login frequency", "group": "Engagement", "unit": "logins / month"},
    "Session_Duration_Avg": {"label": "Avg. session duration", "group": "Engagement", "unit": "minutes"},
    "Pages_Per_Session": {"label": "Pages per session", "group": "Engagement"},
    "Mobile_App_Usage": {"label": "Mobile app usage", "group": "Engagement", "unit": "%"},
    "Social_Media_Engagement_Score": {"label": "Social engagement", "group": "Engagement", "unit": "score"},
    "Email_Open_Rate": {"label": "Email open rate", "group": "Engagement", "unit": "%"},
    "Total_Purchases": {"label": "Total purchases", "group": "Commerce"},
    "Average_Order_Value": {"label": "Average order value", "group": "Commerce", "unit": "$"},
    "Cart_Abandonment_Rate": {"label": "Cart abandonment", "group": "Commerce", "unit": "%"},
    "Wishlist_Items": {"label": "Wishlist items", "group": "Commerce"},
    "Discount_Usage_Rate": {"label": "Discount usage", "group": "Commerce", "unit": "%"},
    "Returns_Rate": {"label": "Returns rate", "group": "Commerce", "unit": "%"},
    "Product_Reviews_Written": {"label": "Reviews written", "group": "Commerce"},
    "Payment_Method_Diversity": {"label": "Payment methods", "group": "Commerce"},
    "Customer_Service_Calls": {"label": "Customer service calls", "group": "Value & support"},
    "Days_Since_Last_Purchase": {"label": "Days since last purchase", "group": "Value & support", "unit": "days"},
    "Lifetime_Value": {"label": "Lifetime value", "group": "Value & support", "unit": "$"},
    "Credit_Balance": {"label": "Credit balance", "group": "Value & support", "unit": "$"},
}


def pretty_feature(name: str) -> str:
    if name.startswith("Gender_"):
        return f"Gender: {name.split('_', 1)[1]}"
    if name.startswith("Signup_Quarter_"):
        return f"Signup: {name.split('_', 2)[2]}"
    if name.startswith("Country_"):
        return f"Country: {name.split('_', 1)[1]}"
    return PRETTY_NAMES.get(name, name.replace("_", " "))


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].astype(int)

    # Missing-value imputation (matches notebook).
    null_cols = X.columns[X.isna().sum() > 1].tolist()
    imputer = SimpleImputer(strategy="median")
    X[null_cols] = imputer.fit_transform(X[null_cols])

    # Scale numeric columns.
    num_cols = X.select_dtypes(include="number").columns.tolist()
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # One-hot encode categoricals.
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(X[CATEGORICAL])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(), index=X.index
    )
    X = pd.concat(
        [X.drop(columns=CATEGORICAL + DROPPED), encoded_df],
        axis=1,
    )

    # Store readable metadata before splitting.
    input_columns = [c for c in df.columns if c != TARGET]
    medians = df[input_columns].median(numeric_only=True)
    modes = {
        col: df[col].mode(dropna=True).iloc[0]
        for col in CATEGORICAL
    }

    defaults: dict[str, object] = {}
    ranges: dict[str, dict[str, float]] = {}
    for col in input_columns:
        if col in CATEGORICAL:
            defaults[col] = str(modes[col])
        elif col in medians.index:
            defaults[col] = round(float(medians[col]), 2)
        else:
            defaults[col] = str(df[col].mode(dropna=True).iloc[0])

    for col in num_cols:
        q = df[col].quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        ranges[col] = {
            "min": round(float(q.loc[0.01]), 2),
            "q25": round(float(q.loc[0.25]), 2),
            "median": round(float(q.loc[0.5]), 2),
            "q75": round(float(q.loc[0.75]), 2),
            "max": round(float(q.loc[0.99]), 2),
            "raw_min": round(float(df[col].min()), 2),
            "raw_max": round(float(df[col].max()), 2),
        }

    categories = {
        col: sorted(str(v) for v in df[col].dropna().unique().tolist())
        for col in CATEGORICAL
    }

    # Same split as the notebook.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=None
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    sorted_proba = np.sort(y_proba)
    risk_percentiles = [
        round(float(np.quantile(sorted_proba, p / 100)), 4) for p in range(1, 100)
    ]

    def json_sample(row: pd.Series) -> dict[str, object]:
        sample: dict[str, object] = {}
        for col in input_columns:
            value = row[col]
            if pd.isna(value):
                sample[col] = None
            elif col in CATEGORICAL or not pd.api.types.is_numeric_dtype(df[col]):
                sample[col] = str(value)
            else:
                sample[col] = round(float(value), 4)
        return sample

    high_risk_sample = json_sample(
        df.loc[X_test.index[int(np.argmax(y_proba))]]
    )
    low_risk_sample = json_sample(
        df.loc[X_test.index[int(np.argmin(y_proba))]]
    )

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "test_size": int(len(y_test)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # ROC curve (compressed to ~60 points for a lightweight payload).
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    step = max(1, len(fpr) // 60)
    roc_points = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), step)
    ]
    if roc_points[-1] != {"fpr": 1.0, "tpr": 1.0}:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})

    # Risk distribution from the held-out test set.
    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(y_proba, bins=bins)
    risk_histogram = [
        {
            "label": f"{int(bins[i] * 100)}–{int(bins[i + 1] * 100)}",
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    # Feature importances (top 14, plus grouped originals).
    importances = sorted(
        (
            {"name": name, "label": pretty_feature(name), "importance": round(float(value), 5)}
            for name, value in zip(model.feature_names_in_, model.feature_importances_)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    grouped: dict[str, float] = {}
    for name, value in zip(model.feature_names_in_, model.feature_importances_):
        base = name.split("_")[0] if any(
            name.startswith(prefix) for prefix in ("Gender_", "Country_", "Signup_Quarter_")
        ) else name
        grouped[base] = grouped.get(base, 0.0) + float(value)
    grouped_importance = sorted(
        (
            {"label": pretty_feature(base), "importance": round(value, 5)}
            for base, value in grouped.items()
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )

    # Overview statistics for the dashboard.
    total = len(df)
    churned = int(y.sum())
    retained = total - churned
    churn_rate = churned / total

    def rate_table(series: pd.Series) -> list[dict[str, object]]:
        table = (
            df.groupby(series, observed=True)[TARGET]
            .agg(["mean", "count"])
            .reset_index()
        )
        table.columns = ["label", "rate", "customers"]
        table = table.sort_values("rate", ascending=False)
        return [
            {
                "label": str(row["label"]),
                "rate": round(float(row["rate"]), 4),
                "customers": int(row["customers"]),
            }
            for _, row in table.iterrows()
        ]

    churn_by_country = rate_table(df["Country"])

    tenure_bins = [0, 1, 2, 3, 5, 10.01]
    tenure_labels = ["<1 yr", "1–2 yrs", "2–3 yrs", "3–5 yrs", "5+ yrs"]
    tenure_series = pd.cut(df["Membership_Years"], bins=tenure_bins, labels=tenure_labels, right=False)
    tenure_table = (
        df.groupby(tenure_series, observed=True)[TARGET]
        .agg(["mean", "count"])
        .reindex(tenure_labels)
        .fillna(0)
    )
    churn_by_tenure = [
        {"label": label, "rate": round(float(row["mean"]), 4), "customers": int(row["count"])}
        for label, row in tenure_table.iterrows()
    ]

    login_bins = [-0.01, 5, 10, 15, 20, 50]
    login_labels = ["0–5", "6–10", "11–15", "16–20", "21+"]
    login_series = pd.cut(df["Login_Frequency"], bins=login_bins, labels=login_labels)
    login_table = (
        df.groupby(login_series, observed=True)[TARGET]
        .agg(["mean", "count"])
        .reindex(login_labels)
        .fillna(0)
    )
    churn_by_login = [
        {"label": label, "rate": round(float(row["mean"]), 4), "customers": int(row["count"])}
        for label, row in login_table.iterrows()
    ]

    support_bins = [-1, 3, 5, 7, 9, 100]
    support_labels = ["0–3", "4–5", "6–7", "8–9", "10+"]
    support_series = pd.cut(df["Customer_Service_Calls"], bins=support_bins, labels=support_labels)
    support_table = (
        df.groupby(support_series, observed=True)[TARGET]
        .agg(["mean", "count"])
        .reindex(support_labels)
        .fillna(0)
    )
    churn_by_support = [
        {"label": label, "rate": round(float(row["mean"]), 4), "customers": int(row["count"])}
        for label, row in support_table.iterrows()
    ]

    stats = {
        "overview": {
            "total_customers": total,
            "churned": churned,
            "retained": retained,
            "churn_rate": round(float(churn_rate), 4),
            "retention_rate": round(float(1 - churn_rate), 4),
            "avg_lifetime_value": round(float(df["Lifetime_Value"].mean()), 2),
            "avg_tenure": round(float(df["Membership_Years"].mean()), 2),
            "avg_order_value": round(float(df["Average_Order_Value"].mean()), 2),
        },
        "churn_by_country": churn_by_country,
        "churn_by_tenure": churn_by_tenure,
        "churn_by_login": churn_by_login,
        "churn_by_support": churn_by_support,
        "risk_histogram": risk_histogram,
        "feature_importance": importances[:14],
        "grouped_importance": grouped_importance[:10],
        "model_comparison": NOTEBOOK_RESULTS,
        "roc": roc_points,
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    bundle = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
        "target": TARGET,
        "null_cols": null_cols,
        "numeric_cols": num_cols,
        "categorical_cols": CATEGORICAL,
        "dropped_cols": DROPPED,
        "input_columns": input_columns,
        "feature_columns": list(X.columns),
        "defaults": defaults,
        "ranges": ranges,
        "categories": categories,
        "field_help": FIELD_HELP,
        "metrics": metrics,
        "stats": stats,
        "samples": {"high_risk": high_risk_sample, "low_risk": low_risk_sample},
        "risk_percentiles": risk_percentiles,
        "threshold": 0.5,
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }

    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    print(f"Saved bundle -> {BUNDLE_PATH}")
    print(json.dumps(metrics, indent=2))
    print("Top features:", [item["label"] for item in importances[:6]])
    print("Risk bands:", risk_histogram)


if __name__ == "__main__":
    main()
