# 🏠 Iron County Home Valuation Engine

A machine-learning-powered property valuation platform built for **Math 4810 — Capstone**.

This application allows authorized users to input property characteristics (or perform a parcel lookup) and receive a professional market value estimate using **two independently trained valuation models**:

- **Linear Model (Lasso → OLS)** for transparency and explainability
- **Non-Linear Model (LightGBM)** for maximum predictive accuracy

Each valuation includes:

- Estimated market value
- 95% prediction interval
- Feature-by-feature contribution breakdown
- SHAP explainability (LightGBM)
- Diagnostic residual analysis (Linear Model)
- Comparable property recommendations (via parcel lookup)
- Full valuation history tracking

---

## 🌐 Live Deployment

**Production App:**  
https://iron-county-home-valuation-model.onrender.com

---

# 📚 Table of Contents

1. Project Overview  
2. Why Two Models?  
3. Full Model Breakdown  
4. Input Fields  
5. Parcel ID Lookup System  
6. Understanding Results  
7. Running Locally  
8. Deploying to Render  
9. Project Structure  
10. Known Limitations  

---

# Project Overview

The Home Valuation Engine is a secure, password-protected web application designed for residential and multi-family property valuation in **Iron County, Utah**.

Users can:

1. Log in securely
2. Enter property details manually or through Parcel ID lookup
3. Choose between two valuation models
4. Receive an estimated market value instantly
5. Review confidence ranges and feature-level explanations
6. Compare against similar sold properties
7. Save and revisit historical valuations

This is not a toy calculator.

It is designed to support real appraisal workflows where accuracy, defensibility, and explanation all matter.

---

# 🎯 What Makes This Different?

Most valuation tools optimize for speed.

This system optimizes for:

## Accuracy  
## Explainability  
## Real-World Validation

That matters more.

---

# Why Two Models?

Because one model is rarely enough.

Highly accurate models are often difficult to explain.

Highly explainable models are often less accurate.

Appraisers need both.

This system provides two separate approaches so users can choose based on the situation.

Sometimes you need the best estimate.

Sometimes you need to defend that estimate to another human.

Those are not always the same problem.

---

# Full Model Breakdown

---

# Linear Model  
## Lasso → OLS Regression

### Performance

- **Test R²:** ~0.88
- **Dollar MAE:** ~$40,000

---

## Step 1 — Feature Selection with Lasso

Lasso regression performs automatic feature selection by shrinking weak coefficients to zero.

This helps remove noise from a large engineered feature space containing:

- polynomial terms (`GLA²`, `Quality²`)
- interaction terms (`Residential × GLA`)
- segment-specific valuation effects

Instead of manually choosing variables, Lasso forces the model to justify every feature.

That is how regression should work.

---

## Step 2 — Final OLS Regression

The surviving variables are passed into:

# Ordinary Least Squares (OLS)

This produces:

- unbiased coefficients
- standard errors
- t-statistics
- p-values

This makes the model fully transparent and highly defensible.

You can explain exactly why the price moved.

That matters when someone challenges the number.

---

## Feature Transformations

Several heavy-tailed predictors are transformed for stability:

### arcsinh transformation

Used for:

- Acreage
- Garage Area
- Basement Sqft
- Total Assessed Value

### Log transformation

Used for:

- Gross Living Area

### Target Variable

The model predicts:

# log(Trended Price)

and then converts predictions back to dollars using exponentiation.

This creates a log-linear valuation system where coefficients represent proportional price effects.

---

## Strengths

- Fully explainable
- Stable and conservative
- Easy to defend to clients
- Supports statistical interpretation
- Strong for standard residential properties

---

## Weaknesses

- Limited ability to model non-linear relationships
- Can miss complex interactions
- Less accurate on unusual or extreme properties

---

# Non-Linear Model  
## LightGBM

### Performance

- **Test R²:** ~0.95
- **Dollar MAE:** ~$20,000

This is the highest-performing model in the system.

By a lot.

---

## Why It Performs Better

LightGBM uses gradient-boosted decision trees.

That allows it to:

- capture non-linear relationships
- model feature interactions automatically
- perform better on high-value properties
- leverage Total Assessed Value more effectively
- avoid assumptions about linearity

Trees do not care about textbook assumptions.

They care about predictive performance.

---

## Hyperparameter Tuning

Hyperparameters were tuned using:

# RandomizedSearchCV

Final selected parameters:

| Parameter | Value |
|---|---:|
| num_leaves | 45 |
| learning_rate | 0.1211 |
| n_estimators | 672 |
| min_child_samples | 13 |
| subsample | 0.9138 |
| colsample_bytree | 0.7861 |
| reg_lambda | 1.4174 |

The target variable is also:

# log(Trended Price)

with predictions exponentiated back into dollars.

---

# Explainability with SHAP

Tree models are powerful.

They are also black boxes.

That is a problem.

To fix that, this system uses:

# SHAP  
## SHapley Additive exPlanations

SHAP provides mathematically exact feature attribution by showing how each feature moves the prediction from the baseline average to the final estimate.

The results page includes:

- contribution table
- waterfall chart
- baseline-to-prediction explanation path

This is not fake interpretability.

It is rigorous.

---

## Strengths

- Highest predictive accuracy
- Handles complex interactions
- Better for expensive and unusual properties
- Strong use of assessor value
- Exact contribution decomposition with SHAP

---

## Weaknesses

- Less intuitive than regression coefficients
- Harder to explain to non-technical users
- Can be overconfident outside training boundaries

---

# Which Model Should You Use?

| Situation | Recommended Model |
|---|---|
| Maximum accuracy matters most | Non-Linear |
| You must explain the estimate clearly | Linear |
| Property is unusual or extreme | Linear |
| Property is a typical residential home | Either |
| You need p-values or coefficient confidence | Linear |
| Property is high-value | Non-Linear |

## Best Practice

Run both.

If both models agree closely, confidence increases.

If they disagree heavily, pay attention.

That usually means the property is unusual, risky, or sitting outside the comfortable boundaries of the training data.

That is where professional judgment matters.

Not blind trust in a model.

---

# Input Fields

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
| Property Type | Example: Single Family Res | No |
| Total Assessed Value | Assessor valuation | Yes |

---

# Important Note

## Total Assessed Value

This is often the single strongest predictor for high-value properties.

It includes:

- land value
- improvements
- assessor prior valuation

Ignoring it is usually a mistake.

Parcel lookup fills this automatically.

Use it.

---

# Parcel ID Lookup System

The system includes a full Parcel ID autofill workflow.

Users can:

1. Upload a parcel CSV
2. Search by parcel number
3. Automatically populate all fields instantly

This includes:

- property segment
- living area
- acreage
- quality score
- garage data
- Total Assessed Value
- comparable properties

This saves time and reduces bad manual input.

Bad inputs create bad outputs.

That should not need explanation.

---

# Parcel CSV File

A example file with 10 random houses features and parcel id is included in the github.

The original parcel data file is **not included** in this repository.

It contains sensitive assessor information and must be uploaded manually each session using:

# UPLOAD CSV

from the application banner.

## Important

The uploaded file is stored:

# In Memory Only

It is:

- not written to disk
- not stored in the database
- lost after server restart

You must re-upload it after deployment restarts.

That is intentional.

---

# Required CSV Columns

The CSV should contain:

| Column | Purpose |
|---|---|
| ParcelNumber | Parcel lookup identifier |
| segment | Residential / Multi-Family |
| TotGLA | Gross living area |
| EffYearBuilt_Weighted | Effective year built |
| Acreage | Lot size |
| FullBaths_Total | Full bathrooms |
| HalfBaths_Total | Half bathrooms |
| Tot Bsmt | Basement square footage |
| GarageArea | Garage size |
| GarageCapacity | Garage capacity |
| Quality_Weighted | Quality score |
| Total Value | Total assessed value |
| Sold Price | Actual sold price |

Column names are normalized automatically.

Examples:

- `Total Value`
- `total_value`

Both work.

Dollar-formatted values like:

`$320,000.00`

are also parsed automatically.

Missing fields do not break lookup.

They simply remain blank for manual entry.

---

# Comparable Property Engine

When parcel lookup is used, the system automatically identifies:

# 5 Most Similar Properties

using:

# K-Nearest Neighbors (KNN)

Similarity is calculated using z-score normalized features such as:

- GLA
- acreage
- quality
- effective year built
- bathrooms
- garage size
- total value

The table displays:

- parcel ID
- GLA
- year built
- acreage
- quality
- actual sold price

This gives users a real-world market sanity check alongside model predictions.

Because if your model says one thing and the market says another, the market wins.

---

# Understanding Results

Every valuation includes more than a number.

---

# Predicted Price

The large result shown at the top is the estimated:

# Current Market Value

This is based on:

# Trended Price

which adjusts historical sales for market appreciation.

It is not simply the historical sold price.

It reflects what the property would likely sell for under current market conditions.

---

# 95% Prediction Interval

Example:

```text
$285,000 – $410,000
```

This reflects where the actual sale price is likely to fall.

This is a:

# Prediction Interval

not a:

# Confidence Interval



---

## Why It Matters

Prediction intervals are computed using:

- held-out test residuals
- grouped by neighborhood (`NbhdCode2`)

Neighborhoods with less data produce wider intervals.

---

# Feature Contribution Table

Both models include a contribution table showing how each feature affected the estimate.

---

## Linear Model

Contributions show:

- how much each feature moves price
- relative to the neighborhood median baseline

Positive values increase price.

Negative values reduce price.

Related terms are grouped into clean business concepts such as:

- GLA
- Quality
- Bathrooms

---

## Non-Linear Model

Contributions are:

# SHAP Values

converted to dollar terms.

They:

- are mathematically exact
- sum directly to the final prediction
- show the true model behavior

The Value column displays the raw input value used by the model.

---

# SHAP Waterfall Chart

Available for the Non-Linear model.

The chart builds value from:

# Baseline → Final Prediction

It shows:

- baseline average price
- positive contributors
- negative contributors
- final predicted price
- prediction interval overlay

---

# Diagnostic Plots

Available for the Linear model.

Four diagnostic plots are shown from held-out test residuals.

| Plot | What You Want |
|---|---|
| Residuals vs Fitted | Random scatter around zero |
| Residuals by Segment | Similar spread |
| Feature Influence | Waterfall Plot |
| Actual vs Predicted | Close to 45° line |

These plots answer one question:

Can we trust the regression assumptions?

---

# Running Locally

---

# Prerequisites

- Python 3.10+
- Git
- `linear_model.pkl` - Included in the Github
- `nonlinear_model.pkl` - Included in the Github

---

# Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

Sqlite:
```env
DATABASE_URL=sqlite:///./your-db-name.db
SESSION_SECRET=your-secret-key-here
```
Other:

```env
DATABASE_URL=your-database-external-url
SESSION_SECRET=your-secret-key-here
```
---

# Adding Users

There is intentionally:

# No Public Registration

Users are created manually using:

```bash
python seed_user.py <username> <email> <password>
```

Example:

```bash
python seed_user.py admin admin@example.com mypassword
```
---

# Starting the App

Production-style run:

```bash
fastapi run main.py
```

Development with auto reload:

```bash
fastapi dev main.py
```

Then open:

http://localhost:8000

and log in.

If it does not start, check your `.env` to ensure your databse is being called properly.
---

# Project Structure

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

Clean structure matters.

Messy projects create messy developers.

---

# Known Limitations

---

# Residential + Multi-Family Only

Land sales were excluded.

Vacant land valuation behaves differently and both models perform poorly there.

Different asset class.

Different problem.

---

# Iron County Only

The model is trained specifically for:

# Iron County, Utah

Using it elsewhere is statistically irresponsible.

Do not do it.

---

# No Live Market Feed

Predictions rely on trended historical sales.

This system does not consume live MLS pricing.

It is not magic.

---

# Parcel CSV Is Session-Based

Uploaded parcel files are stored only in memory.

Server restart means:

gone.

Re-upload required.

---

# Comparable Properties Require Lookup

Comparable property search only works when parcel lookup is used.

Manual entry alone does not trigger KNN comparables.

That is by design.

---

# No Public Registration

Accounts must be created manually.

This is intentional.

The system is designed for controlled professional use, not open public access.

Good systems are often restrictive on purpose.

---

# Final Note

If both models agree:

good.

If both models disagree:

pay attention.
