"""ChurnLens — FastAPI backend for the e-commerce churn model.

Run:
    uvicorn server:app --reload --port 8000

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import bisect
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without adding a dotenv dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


BASE_DIR = Path(__file__).resolve().parent
load_env_file(BASE_DIR / ".env")

UI_DIR = BASE_DIR / "UI"
BUNDLE_PATH = BASE_DIR / os.getenv("MODEL_BUNDLE", "churn_model_bundle.joblib")
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

GROUP_ORDER = ["Profile", "Engagement", "Commerce", "Value & support"]
INTEGER_FIELDS = {
    "Age",
    "Wishlist_Items",
    "Total_Purchases",
    "Customer_Service_Calls",
    "Product_Reviews_Written",
    "Payment_Method_Diversity",
}
UNITS = {
    "Age": "yrs",
    "Membership_Years": "yrs",
    "Login_Frequency": "logins/mo",
    "Session_Duration_Avg": "min",
    "Pages_Per_Session": "pages",
    "Cart_Abandonment_Rate": "%",
    "Discount_Usage_Rate": "%",
    "Returns_Rate": "%",
    "Email_Open_Rate": "%",
    "Social_Media_Engagement_Score": "pts",
    "Mobile_App_Usage": "%",
    "Average_Order_Value": "$",
    "Lifetime_Value": "$",
    "Credit_Balance": "$",
    "Days_Since_Last_Purchase": "days",
}


class CustomerInput(BaseModel):
    """A customer snapshot. Any omitted field is filled with its dataset default."""

    Age: float | None = Field(None, ge=0, description="Customer age in years")
    Gender: str | None = Field(None, description="Female, Male, or Other")
    Country: str | None = Field(None, description="Customer country")
    City: str | None = Field(None, description="Customer city (not used by the model)")
    Membership_Years: float | None = Field(None, ge=0, description="Years as a member")
    Login_Frequency: float | None = Field(None, ge=0, description="Logins per month")
    Session_Duration_Avg: float | None = Field(None, ge=0, description="Average session minutes")
    Pages_Per_Session: float | None = Field(None, ge=0, description="Pages viewed per session")
    Cart_Abandonment_Rate: float | None = Field(None, ge=0, description="Cart abandonment rate (%)")
    Wishlist_Items: float | None = Field(None, ge=0, description="Items on wishlist")
    Total_Purchases: float | None = Field(None, description="Lifetime purchase count")
    Average_Order_Value: float | None = Field(None, ge=0, description="Average order value ($)")
    Days_Since_Last_Purchase: float | None = Field(None, ge=0, description="Recency in days")
    Discount_Usage_Rate: float | None = Field(None, ge=0, description="Share of discounted orders (%)")
    Returns_Rate: float | None = Field(None, ge=0, description="Return rate (%)")
    Email_Open_Rate: float | None = Field(None, ge=0, description="Email open rate (%)")
    Customer_Service_Calls: float | None = Field(None, ge=0, description="Support calls made")
    Product_Reviews_Written: float | None = Field(None, ge=0, description="Reviews written")
    Social_Media_Engagement_Score: float | None = Field(None, ge=0, description="Social engagement score")
    Mobile_App_Usage: float | None = Field(None, ge=0, description="Mobile app usage (%)")
    Payment_Method_Diversity: float | None = Field(None, ge=0, description="Distinct payment methods")
    Lifetime_Value: float | None = Field(None, ge=0, description="Lifetime value ($)")
    Credit_Balance: float | None = Field(None, ge=0, description="Credit balance ($)")
    Signup_Quarter: str | None = Field(None, description="Q1, Q2, Q3, or Q4")


class PredictionResponse(BaseModel):
    probability: float
    percent: float
    prediction: int
    label: str
    risk_band: str
    confidence: float
    percentile: int
    value_at_risk: float
    lifetime_value: float
    threshold: float
    factors: list[dict[str, Any]]
    recommendations: list[str]


def build_fields(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    categories = bundle["categories"]
    ranges = bundle["ranges"]
    defaults = bundle["defaults"]
    help_map = bundle["field_help"]

    for col in bundle["input_columns"]:
        if col in bundle["dropped_cols"]:
            continue

        meta = help_map.get(col, {})
        label = meta.get("label", col.replace("_", " "))
        group = meta.get("group", "Other")
        unit = UNITS.get(col) or meta.get("unit")
        value = defaults.get(col)

        if col in categories:
            fields.append(
                {
                    "name": col,
                    "label": label,
                    "group": group,
                    "type": "select",
                    "unit": None,
                    "default": str(value),
                    "options": categories[col],
                }
            )
            continue

        rng = ranges.get(col, {})
        raw_min = rng.get("raw_min", 0)
        # Keep inputs permissive: allow zero unless the source data contains
        # genuinely negative values (e.g. Total_Purchases).
        input_min = raw_min if raw_min < 0 else 0
        step = 1 if col in INTEGER_FIELDS else 0.01 if col == "Average_Order_Value" else 0.1
        fields.append(
            {
                "name": col,
                "label": label,
                "group": group,
                "type": "number",
                "unit": unit,
                "default": value,
                "min": input_min,
                "max": rng.get("raw_max", None),
                "median": rng.get("median"),
                "q25": rng.get("q25"),
                "q75": rng.get("q75"),
                "step": step,
            }
        )

    fields.sort(key=lambda item: (GROUP_ORDER.index(item["group"]) if item["group"] in GROUP_ORDER else 99))
    return fields


def risk_band(probability: float) -> tuple[str, str]:
    if probability < 0.35:
        return "Low", "low"
    if probability < 0.65:
        return "Elevated", "medium"
    return "High", "high"


def format_value(value: Any, unit: str | None, is_integer: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if is_integer or abs(number - round(number)) < 1e-9:
        text = f"{int(round(number)):,}"
    elif abs(number) >= 1000:
        text = f"{number:,.0f}"
    else:
        text = f"{number:,.1f}"
    if not unit:
        return text
    if unit == "$":
        return f"${text}"
    if unit == "%":
        return f"{text}%"
    return f"{text} {unit}"


def prepare_frame(bundle: dict[str, Any], raw: dict[str, Any]) -> pd.DataFrame:
    values: dict[str, Any] = {}
    for col in bundle["input_columns"]:
        value = raw.get(col)
        if value is None or (isinstance(value, str) and not value.strip()):
            value = bundle["defaults"].get(col)
        values[col] = value

    frame = pd.DataFrame([values], columns=bundle["input_columns"])
    for col in bundle["numeric_cols"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame[bundle["null_cols"]] = bundle["imputer"].transform(frame[bundle["null_cols"]])
    frame[bundle["numeric_cols"]] = bundle["scaler"].transform(frame[bundle["numeric_cols"]])

    encoded = bundle["encoder"].transform(frame[bundle["categorical_cols"]])
    encoded_df = pd.DataFrame(
        encoded,
        columns=bundle["encoder"].get_feature_names_out(),
        index=frame.index,
    )
    model_frame = pd.concat(
        [frame.drop(columns=bundle["categorical_cols"] + bundle["dropped_cols"]), encoded_df],
        axis=1,
    )
    return model_frame[bundle["feature_columns"]]


def predict_probability(bundle: dict[str, Any], raw: dict[str, Any]) -> float:
    frame = prepare_frame(bundle, raw)
    return float(bundle["model"].predict_proba(frame)[0, 1])


def local_factors(
    bundle: dict[str, Any], raw: dict[str, Any], base_probability: float
) -> list[dict[str, Any]]:
    """One-at-a-time impact estimate: how much each signal moves risk
    relative to a typical (median / modal) customer."""
    defaults = bundle["defaults"]
    fields = {field["name"]: field for field in build_fields(bundle)}
    candidates = bundle["numeric_cols"] + bundle["categorical_cols"]
    factors: list[dict[str, Any]] = []

    for col in candidates:
        neutral = defaults.get(col)
        current = raw.get(col, neutral)
        if current is None:
            current = neutral
        try:
            same = abs(float(current) - float(neutral)) < 1e-9
        except (TypeError, ValueError):
            same = str(current) == str(neutral)
        if same:
            continue

        modified = dict(raw)
        modified[col] = neutral
        try:
            without = predict_probability(bundle, modified)
        except Exception:
            continue
        impact = (base_probability - without) * 100

        meta = fields.get(col, {})
        factors.append(
            {
                "label": meta.get("label", col.replace("_", " ")),
                "impact": round(impact, 1),
                "direction": "up" if impact >= 0 else "down",
                "value": format_value(current, meta.get("unit"), col in INTEGER_FIELDS),
                "typical": format_value(neutral, meta.get("unit"), col in INTEGER_FIELDS),
            }
        )

    factors.sort(key=lambda item: abs(item["impact"]), reverse=True)
    factors = [factor for factor in factors if abs(factor["impact"]) >= 0.2][:5]

    if not factors:
        for item in bundle["stats"]["grouped_importance"][:3]:
            factors.append(
                {
                    "label": item["label"],
                    "impact": 0,
                    "direction": "neutral",
                    "value": "—",
                    "typical": "—",
                }
            )
    return factors


def build_recommendations(
    bundle: dict[str, Any], raw: dict[str, Any], probability: float
) -> list[str]:
    ranges = bundle["ranges"]
    defaults = bundle["defaults"]

    def value(col: str) -> float:
        current = raw.get(col, defaults.get(col))
        try:
            return float(current)
        except (TypeError, ValueError):
            return float(defaults.get(col, 0))

    def above_q75(col: str) -> bool:
        q75 = (ranges.get(col) or {}).get("q75")
        return q75 is not None and value(col) >= q75

    def below_q25(col: str) -> bool:
        q25 = (ranges.get(col) or {}).get("q25")
        return q25 is not None and value(col) <= q25

    recommendations: list[str] = []
    if above_q75("Days_Since_Last_Purchase"):
        recommendations.append(
            "Launch a win-back sequence with a personalised offer — recency is the strongest live signal for this customer."
        )
    if above_q75("Customer_Service_Calls"):
        recommendations.append(
            "Route this account to priority support and follow up on the open issue before the next billing cycle."
        )
    if above_q75("Cart_Abandonment_Rate"):
        recommendations.append(
            "Trigger a cart-recovery message with a time-boxed incentive and a clear checkout path."
        )
    if above_q75("Discount_Usage_Rate"):
        recommendations.append(
            "Protect margin with loyalty pricing instead of blanket discounts; reinforce the value they already pay for."
        )
    if below_q25("Login_Frequency") or below_q25("Email_Open_Rate"):
        recommendations.append(
            "Re-engage with a habit-building email or in-app nudge featuring the categories they browse most."
        )
    if above_q75("Returns_Rate"):
        recommendations.append(
            "Check in on product fit and sizing — elevated returns often precede a silent exit."
        )

    if probability >= 0.65 and len(recommendations) < 2:
        recommendations.append(
            "Schedule a personal check-in within the next seven days — the score is high enough that waiting for a campaign cycle could be too late."
        )
    if probability < 0.35:
        recommendations.append(
            "Keep this customer in the healthy cohort: nurture with loyalty perks and early access, not retention discounts."
        )
    if not recommendations:
        recommendations.append(
            "A lightweight check-in and a relevant product recommendation are enough to keep this customer warm."
        )
    return recommendations[:3]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bundle = joblib.load(BUNDLE_PATH)
    except Exception as exc:
        # Handles a missing bundle or a version mismatch between the saved
        # scikit-learn objects and the runtime. The CSV is the source of truth.
        print(f"Could not load {BUNDLE_PATH.name} ({exc}); rebuilding it now…")
        try:
            from train_model import main as build_bundle

            build_bundle()
            bundle = joblib.load(BUNDLE_PATH)
        except Exception as rebuild_exc:
            raise RuntimeError(
                "churn_model_bundle.joblib is missing or unreadable, and automatic "
                "rebuilding failed. Run `python train_model.py` and restart the server."
            ) from rebuild_exc

    app.state.bundle = bundle
    app.state.fields = build_fields(bundle)
    yield


app = FastAPI(
    title="ChurnLens API",
    version="1.0.0",
    description="Churn scoring and portfolio insights for the e-commerce dataset.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if UI_DIR.exists():
    app.mount("/assets", StaticFiles(directory=UI_DIR), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    bundle = app.state.bundle
    return {
        "status": "ok",
        "model": "Gradient Boosting",
        "trained_at": bundle["trained_at"],
        "features": len(bundle["feature_columns"]),
    }


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    bundle = app.state.bundle
    return {
        "model": {
            "name": "Gradient Boosting",
            "version": bundle["trained_at"],
            "threshold": bundle["threshold"],
            "metrics": bundle["metrics"],
        },
        "groups": GROUP_ORDER,
        "fields": app.state.fields,
        "samples": bundle["samples"],
        "field_count": len(app.state.fields),
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    bundle = app.state.bundle
    return {"stats": bundle["stats"], "trained_at": bundle["trained_at"]}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> PredictionResponse:
    bundle = app.state.bundle
    raw = customer.model_dump()
    try:
        probability = predict_probability(bundle, raw)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=400, detail=f"Could not score customer: {exc}") from exc

    band, _ = risk_band(probability)
    prediction = int(probability >= bundle["threshold"])
    percentiles = bundle.get("risk_percentiles", [])
    percentile = (
        max(1, round(bisect.bisect_right(percentiles, probability) / len(percentiles) * 100))
        if percentiles
        else 0
    )

    lifetime_value = float(
        raw.get("Lifetime_Value")
        if raw.get("Lifetime_Value") is not None
        else bundle["defaults"].get("Lifetime_Value", 0)
    )
    value_at_risk = probability * lifetime_value

    return PredictionResponse(
        probability=round(probability, 4),
        percent=round(probability * 100, 1),
        prediction=prediction,
        label="Likely to churn" if prediction else "Likely to stay",
        risk_band=band,
        confidence=round(max(probability, 1 - probability), 4),
        percentile=percentile,
        value_at_risk=round(value_at_risk, 2),
        lifetime_value=round(lifetime_value, 2),
        threshold=bundle["threshold"],
        factors=local_factors(bundle, raw, probability),
        recommendations=build_recommendations(bundle, raw, probability),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=APP_HOST, port=APP_PORT, reload=False)
