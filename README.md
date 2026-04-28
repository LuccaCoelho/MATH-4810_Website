# 🏠 Iron County Home Valuation Engine

A machine-learning-powered property valuation platform built for **Math 4810 — Capstone**.

This application allows authorized users to input property characteristics (or perform a parcel lookup) and receive a professional market value estimate using **two independently trained valuation models**:

- **Linear Model (Lasso → OLS)** — optimized for transparency and explainability
- **Non-Linear Model (LightGBM)** — optimized for maximum predictive accuracy

Each valuation includes:

- Estimated market value
- 95% prediction interval
- Feature-by-feature contribution breakdown
- SHAP explainability (LightGBM)
- Diagnostic residual analysis (Linear Model)
- Comparable property recommendations (via parcel lookup)
- Full valuation history tracking

**Production App:** https://iron-county-home-valuation-model.onrender.com

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Why Two Models?](#why-two-models)
3. [Full Model Breakdown](#full-model-breakdown)
4. [Input Fields](#input-fields)
5. [Parcel ID Lookup System](#parcel-id-lookup-system)
6. [Understanding Results](#understanding-results)
7. [Running Locally](#running-locally)
8. [Deploying to Render](#deploying-to-render)
9. [Project Structure](#project-structure)
10. [Known Limitations](#known-limitations)

---

## Project Overview

The Home Valuation Engine is a secure, password-protected web application designed for residential and multi-family property valuation in **Iron County, Utah**.

Users can:

1. Log in securely
2. Enter property details manually or through Parcel ID lookup
3. Choose between two valuation models
4. Receive an estimated market value instantly
5. Review confidence ranges and feature-level explanations
6. Compare against similar sold properties
7. Save and revisit historical valuations

This is not a toy calculator. It is designed to support real appraisal workflows where accuracy, defensibility, and explanation all matter.

---

## 🎯 What Makes This Different?

Most valuation tools optimize for speed. This system optimizes for **accuracy**, **explainability**, and **real-world validation** — and that matters more.

---

## Why Two Models?

Because one model is rarely enough.

Highly accurate models are often difficult to explain. Highly explainable models are often less accurate. Appraisers need both. This system provides two separate approaches so users can choose based on the situation — sometimes you need the best estimate, and sometimes you need to defend that estimate to another human. Those are not always the same problem.

---

## Full Model Breakdown

### Linear Model — Lasso → OLS Regression

**Performance**
- Test R²: ~0.88
- Dollar MAE: ~$34,719

#### Step 1 — Feature Selection with Lasso

Lasso regression performs automatic feature selection by shrinking weak coefficients to zero. This removes noise from a large engineered feature space that includes polynomial terms (`GLA²`, `Quality²`), interaction terms (`Residential × GLA`), and segment-specific valuation effects. Rather than manually choosing variables, Lasso forces the model to justify every feature it keeps.

#### Step 2 — Final OLS Regression

The surviving variables are passed into **Ordinary Least Squares (OLS)**, which produces unbiased coefficients, standard errors, t-statistics, and p-values. This makes the model fully transparent and highly defensible — you can explain exactly why the price moved, which matters when someone challenges the number.

#### Feature Transformations

Several heavy-tailed predictors are transformed for stability:

| Transformation | Applied To |
|---|---|
| arcsinh | Acreage, Garage Area, Basement Sqft, Total Assessed Value |
| log | Gross Living Area |

The target variable is `log(Trended Price)`, with predictions converted back to dollars via exponentiation. This produces a log-linear system where coefficients represent proportional price effects.

#### Strengths
- Fully explainable — coefficients have direct interpretable meaning
- Stable and conservative
- Easy to defend to clients and stakeholders
- Supports formal statistical interpretation
- Strong for standard residential properties

#### Weaknesses
- Limited ability to model non-linear relationships
- Can miss complex feature interactions
- Less accurate on unusual or extreme properties

---

### Non-Linear Model — LightGBM

**Performance**
- Test R²: ~0.97
- Dollar MAE: ~$12,239

This is the highest-performing model in the system.

#### Why It Performs Better

LightGBM uses gradient-boosted decision trees, which allows it to capture non-linear relationships, model feature interactions automatically, perform better on high-value properties, leverage Total Assessed Value more effectively, and avoid assumptions about linearity. Trees optimize for predictive performance without requiring textbook statistical assumptions.

#### Explainability with SHAP

Tree models are powerful — but also opaque. To address this, the system uses **SHAP (SHapley Additive exPlanations)**, which provides mathematically exact feature attribution by showing how each feature moves the prediction from the baseline average to the final estimate.

The results page includes a contribution table, a waterfall chart, and a full baseline-to-prediction explanation path. This is rigorous interpretability, not a cosmetic approximation.

#### Strengths
- Highest predictive accuracy in the system
- Handles complex interactions and non-linearities
- Better for expensive and unusual properties
- Strong use of assessor value
- Exact contribution decomposition via SHAP

#### Weaknesses
- Less intuitive than regression coefficients
- Harder to explain to non-technical users
- Can be overconfident outside training boundaries

---

## Which Model Should You Use?

| Situation | Recommended Model |
|---|---|
| Maximum accuracy matters most | Non-Linear |
| You must explain the estimate clearly | Linear |
| Property is unusual or extreme | Linear |
| Property is a typical residential home | Either |
| You need p-values or coefficient confidence | Linear |
| Property is high-value (>$700k) | Linear |

**Best practice: run both.** If both models agree closely, confidence increases. If they disagree significantly, pay attention — that usually means the property is unusual, risky, or sitting outside the comfortable boundaries of the training data. That is where professional judgment matters, not blind trust in a model.

---

## Input Fields

Both models use the same property input fields.

| Field | Description | Required |
|---|---|---|
| Property Segment | Residential or Multi-Family | Yes |
| Gross Living Area | Above-grade living area (sq ft) | Yes |
| Effective Year Built | Weighted effective year built | Yes |
| Acreage | Lot size in acres | Yes |
| Quality Score | Weighted quality score (1–5) | Yes |
| Full Baths | Number of full bathrooms | Yes |
| Half Baths | Number of half bathrooms | Yes |
| Basement Sqft | Total basement area | No |
| Garage Area | Garage area (sq ft) | No |
| Garage Capacity | Number of cars | No |
| Property Type | E.g., Single Family Res | No |
| Total Assessed Value | Assessor valuation | Yes |

> **Note on Total Assessed Value:** This is often the single strongest predictor, especially for high-value properties. It incorporates land value, improvements, and the assessor's prior valuation. Parcel lookup fills this automatically — use it.

---

## Parcel ID Lookup System

The system includes a full Parcel ID autofill workflow. Users can upload a parcel CSV, search by parcel number, and automatically populate all input fields instantly — including property segment, living area, acreage, quality score, garage data, Total Assessed Value, and comparable properties. This saves time and reduces manual entry errors. Bad inputs create bad outputs.

### Parcel CSV File

An example CSV with 10 sample properties is included in the repository. The original parcel data file is **not included** — it contains sensitive assessor information and must be uploaded manually each session via the **UPLOAD CSV** button in the application banner.

> ⚠️ **Uploaded files are stored in memory only.** They are not written to disk or stored in the database, and are lost on server restart. You must re-upload after each deployment restart. This is intentional.

### Required CSV Columns

| Column | Purpose |
|---|---|
| ParcelID | Unique parcel identifier |
| Segment | Residential or Multi-Family |
| GLA | Gross Living Area (sq ft) |
| EffYrBlt | Effective Year Built |
| Acreage | Lot size |
| QualScore | Quality score |
| FullBaths | Full bathrooms |
| HalfBaths | Half bathrooms |
| BsmtSqft | Basement square footage |
| GarageArea | Garage area (sq ft) |
| GarageCap | Garage capacity |
| PropType | Property type description |
| TotalValue | Total Assessed Value |
| NbhdCode2 | Neighborhood code |

---

## Comparable Property Engine

When parcel lookup is used, the system automatically identifies the **5 most similar properties** using **K-Nearest Neighbors (KNN)**. Similarity is calculated using z-score normalized features: GLA, acreage, quality, effective year built, bathrooms, garage size, and total value.

The comparison table displays parcel ID, GLA, year built, acreage, quality, and actual sold price — giving users a real-world market sanity check alongside model predictions. If your model says one thing and the market says another, the market wins.

---

## Understanding Results

### Predicted Price

The main result is the estimated **current market value** based on a **trended price** — not the raw historical sold price. Trended price adjusts historical sales for market appreciation, reflecting what the property would likely sell for under current market conditions.

### 95% Prediction Interval

Example: `$285,000 – $410,000`

This is a **prediction interval**, not a confidence interval. It reflects where an individual sale price is likely to fall. Intervals are computed from held-out test residuals grouped by neighborhood (`NbhdCode2`). Neighborhoods with less data produce wider intervals.

### Feature Contribution Table

Both models include a contribution table showing how each input feature affected the final estimate.

**Linear Model:** Contributions show how much each feature moves the price relative to the neighborhood median baseline. Positive values increase price; negative values reduce it. Related terms are grouped into clean business concepts (GLA, Quality, Bathrooms).

**Non-Linear Model:** Contributions are SHAP values converted to dollar terms. They are mathematically exact and sum directly to the final prediction, reflecting the true model behavior.

### SHAP Waterfall Chart *(Non-Linear model only)*

The waterfall chart builds value step-by-step from the baseline average to the final prediction. It displays the baseline price, each positive and negative contributor, and the final predicted price with a prediction interval overlay.

### Diagnostic Plots *(Linear model only)*

Four diagnostic plots are shown from held-out test residuals.

| Plot | What You Want to See |
|---|---|
| Residuals vs Fitted | Random scatter around zero |
| Residuals by Segment | Similar spread across segments |
| Feature Influence | Waterfall chart |
| Actual vs Predicted | Points close to the 45° line |

These plots answer one question: can we trust the regression assumptions?

---

## Running Locally

### Prerequisites

- Python 3.10+
- Git
- `linear_model.pkl` — included in the repository
- `nonlinear_model.pkl` — included in the repository

### Installation

```bash
git clone https://github.com/LuccaCoelho/MATH-4810_Website.git
cd MATH-4810_Website

python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root.

**SQLite (local development):**
```env
DATABASE_URL=sqlite:///./your-db-name.db
SESSION_SECRET=your-secret-key-here
```

**External database (production):**
```env
DATABASE_URL=your-database-external-url
SESSION_SECRET=your-secret-key-here
```

### Adding Users

For this to work, you will need a .db file and the path in the environments file to run locally.

There is no public registration. Users are created manually:

```bash
python seed_user.py <username> <email> <password>
```

Example:
```bash
python seed_user.py admin admin@example.com mypassword
```

### Starting the App

```bash
# Production-style
fastapi run main.py

# Development with auto-reload
fastapi dev main.py
```

Then open http://localhost:8000 and log in.

A demo login was put int the deployed webapp using the random house data example.

**Username**: demo

**Password**: demo123

If the app fails to start, check your `.env` file to ensure the database URL is configured correctly.

---

## Project Structure

```text
.
├── main.py
├── seed_user.py
├── requirements.txt
├── .env
│
├── linear_model.pkl
├── nonlinear_model.pkl
│
├── db/
│   └── database.py
│
├── models/
│   └── models.py
│
├── auth/
│   ├── router.py
│   ├── dependencies.py
│   └── auth.py
│
├── templates/
│   ├── index.html
│   ├── main.html
│   ├── main_nonlinear.html
│   ├── result.html
│   └── history.html
│
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── main.js
│
└── data/
    └── parcel_example.csv
```

---

## Known Limitations

**Residential and Multi-Family only.** Land sales were excluded — vacant land valuation behaves differently and both models perform poorly on it. It's a different asset class and a different problem.

**Iron County, Utah only.** The models were trained specifically on Iron County data. Applying them elsewhere will produce unreliable results.

**No live market feed.** Predictions rely on trended historical sales and do not consume live MLS pricing.

**Parcel CSV is session-based.** Uploaded parcel files are stored in memory only. A server restart clears them, requiring a manual re-upload.

**Comparable properties require parcel lookup.** KNN comparables are only triggered when parcel lookup is used. Manual entry alone does not activate the comparable property engine — this is by design.

**No public registration.** Accounts must be created manually via `seed_user.py`. The system is designed for controlled professional use, not open public access.

### Non-Linear Model — High-Value Property Limitation

The LightGBM model tends to **undervalue properties above approximately $700,000**. This is expected behavior driven by the structure of the training data and the nature of tree-based models.

**Root causes:**

1. **Skewed training distribution.** The dataset is concentrated around mid-range values (~$400k). High-value properties are relatively rare, so the model is optimized for the majority of cases and exhibits downward bias on expensive homes.

2. **No extrapolation.** Gradient-boosted trees cannot extrapolate beyond observed training ranges. Predictions are formed from averages within decision tree leaves, so the model can't confidently predict values well above those in training.

3. **Averaging within leaves.** High-value homes are often grouped with moderately high homes in the same decision regions, leading to systematic underestimation through averaging effects.

4. **Regularization.** The model's regularization (learning rate and L2 penalty) reduces overfitting but also dampens extreme predictions, contributing to conservative estimates at the upper end of the market.

**For properties above ~$700k:** expect the Non-Linear model to skew low, expect more disagreement between models, and treat the Linear model as the more reliable reference. Run both, compare carefully, and use comparable properties as a validation check.

---

## Final Note

If both models agree — good. If they disagree — pay attention.
