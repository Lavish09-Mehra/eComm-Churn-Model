# ChurnLens

A clean churn-intelligence dashboard for the e-commerce dataset in this folder. It pairs a tuned
scikit-learn Gradient Boosting model with a FastAPI backend and a dependency-free HTML/CSS/JS UI.

**Links:** [GitHub — @Lavish09-Mehra](https://github.com/Lavish09-Mehra) ·
[Repository — eComm-Churn-Model](https://github.com/Lavish09-Mehra/eComm-Churn-Model)

## What is included

- **FastAPI backend** — loads a packaged model bundle and exposes prediction + portfolio-insight endpoints.
- **Churn scoring** — probability, risk band, percentile, value at risk, local factor impacts, and retention recommendations.
- **Dashboard UI** — dark/light mode, local fonts (Manrope + Space Grotesk), smooth scrolling, reveal-on-scroll motion, and no gradients.
- **Charts** — churn distribution, churn by country, service calls vs. churn, predicted-risk histogram, feature importance, ROC curve, and a seven-model benchmark.

## Project layout

```
ecomm-model/
├── server.py                  # FastAPI application
├── train_model.py             # Rebuilds the model bundle from the CSV
├── churn_model_bundle.joblib  # Packaged model + preprocessing + dashboard stats
├── eCommerce-dataset.csv      # Source data
├── .env.example               # Environment variable template
├── .gitignore
├── UI/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── fonts/                 # Locally hosted variable fonts
├── churn_model.pkl            # Original notebook artifact (corrupted; kept for reference)
└── churn_guess.ipynb          # Original exploration / training notebook
```

## Run it

```bash
cd ecomm-model
cp .env.example .env
python -m pip install -r requirements.txt
python -m uvicorn server:app --reload --port 8000
```

`.env` is loaded automatically by `server.py` (without an extra dependency). It is intentionally
gitignored; use `.env.example` as the committed template.

Open <http://127.0.0.1:8000>.

If you use the existing Jupyter environment on this machine:

```bash
/home/lavish09/jupyter-env/bin/python -m uvicorn server:app --port 8000
```

### Rebuild the model (optional)

The bundle is already included. To retrain it from the source CSV:

```bash
python train_model.py
```

This mirrors the preprocessing in `churn_guess.ipynb`: median imputation, standard scaling of numeric
columns, one-hot encoding for Gender / Signup_Quarter / Country, and a tuned Gradient Boosting classifier.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Model/status check |
| `GET` | `/api/meta` | Field metadata, defaults, samples, metrics |
| `GET` | `/api/stats` | Aggregated dashboard chart data |
| `POST` | `/api/predict` | Score a customer snapshot |
| `GET` | `/docs` | Interactive OpenAPI docs |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 29, "Gender": "Female", "Country": "USA", "Membership_Years": 1.2,
       "Login_Frequency": 4, "Session_Duration_Avg": 18, "Pages_Per_Session": 5,
       "Cart_Abandonment_Rate": 82, "Wishlist_Items": 2, "Total_Purchases": 5,
       "Average_Order_Value": 81, "Days_Since_Last_Purchase": 64,
       "Discount_Usage_Rate": 38, "Returns_Rate": 12, "Email_Open_Rate": 6,
       "Customer_Service_Calls": 9, "Product_Reviews_Written": 1,
       "Social_Media_Engagement_Score": 8, "Mobile_App_Usage": 7,
       "Payment_Method_Diversity": 2, "Lifetime_Value": 680,
       "Credit_Balance": 950, "Signup_Quarter": "Q2"}'
```

## Links

- GitHub profile: <https://github.com/Lavish09-Mehra>
- Repository: <https://github.com/Lavish09-Mehra/eComm-Churn-Model>
- Clone:

```bash
git clone https://github.com/Lavish09-Mehra/eComm-Churn-Model.git
```

## Model notes

- Algorithm: `GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
  min_samples_leaf=2, random_state=42)`.
- Held-out test performance from the rebuilt bundle: **92.0% accuracy**, **93.2% precision**,
  **78.0% recall**, **84.9% F1**, **0.926 ROC AUC**.
- `churn_model.pkl` from the original notebook could not be loaded (the pickle contains corrupted
  class references), so the model was rebuilt from the notebook pipeline and saved as
  `churn_model_bundle.joblib` with its preprocessing objects and dashboard metadata.
- The data is synthetic; this is a demo, not a production decision system.

---

Built and maintained by [Lavish Mehra](https://github.com/Lavish09-Mehra).
