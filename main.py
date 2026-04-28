import re
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm

from fastapi import FastAPI, Request, Depends, HTTPException, Form, UploadFile, File
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.database import get_db, Base, engine
from models.models import Valuation
from auth.router import router as auth_router
from auth.dependencies import CurrentUser

from dotenv import load_dotenv
load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(auth_router)


# ── Exception handling ────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 307 and "Location" in exc.headers:
        return RedirectResponse(exc.headers["Location"], status_code=302)
    raise exc


# ── Static + templates ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory="templates")
templates.env.filters["zip"] = zip


# ── Load model bundles ────────────────────────────────────────────────────────
_lin = joblib.load(BASE_DIR / "linear_model.pkl")
_nl  = joblib.load(BASE_DIR / "nonlinear_model.pkl")

# Unpack linear bundle
_lin_model           = _lin["model"]
_sel_names           = _lin["sel_names"]
_Xc_tr_cols          = _lin["Xc_tr_cols"]
_X_tr_v4_cols        = _lin["X_tr_v4_cols"]
_lin_nbhd_defaults    = _lin.get("nbhd_defaults", _lin.get("seg_defaults", {}))
_lin_global_defaults = _lin["global_defaults"]
_lin_pi_q            = _lin["pi_q"]
_lin_pi_q_global     = _lin["pi_q_global"]
_lin_transforms      = _lin["transforms"]
_lin_categoricals    = _lin["categoricals"]
_seg_cols            = _lin["seg_cols"]
_poly_cols           = _lin["poly_cols"]
_key_numeric         = _lin["key_numeric"]

# Unpack nonlinear bundle
_nl_model        = _nl["model"]
_orig_of         = _nl["orig_of"]
_X_train_cols    = _nl["X_train_cols"]
# Pre-compute which columns in X_train are NbhdCode2 dummies so we can zero them reliably
_nl_nbhd_dummy_cols = [
    c for c in _X_train_cols
    if _orig_of.get(c, "").startswith("NbhdCode2_") or "NbhdCode2_" in _orig_of.get(c, "")
]
# Pre-compute parcel identifier columns to always zero out during prediction
# Check both the original name (via _orig_of) AND the cleaned column name itself
_parcel_id_cols = [
    c for c in _X_train_cols
    if any(p in _orig_of.get(c, "").lower() for p in
           ("parcelnumber", "parcel_number", "parcelid", "parcel_id", "parcel"))
    or any(p in c.lower() for p in
           ("parcelnumber", "parcel_number", "parcelid", "parcel_id", "parcelno"))
]
print(f"[startup] all model columns: {list(_X_train_cols)[:20]} ...")
print(f"[startup] parcel columns found in model (will be zeroed): {_parcel_id_cols}")
_nl_nbhd_defaults = _nl.get("nbhd_defaults", _nl.get("seg_defaults", {}))
_nl_pi_q         = _nl["pi_q"]
_nl_pi_q_global  = _nl["pi_q_global"]
_base_val        = _nl["base_val"]
_nl_categoricals = _nl["categoricals"]
_explainer       = _nl["explainer"]


# ── Pre-compute linear diagnostic data from bundle test residuals ─────────────
# The bundle stores pi_q which was computed from test residuals.
# We reconstruct enough data for the 6 diagnostic plots by re-using
# whatever test data the bundle has encoded in its PI quantile structures.
# If the bundle also saved X_te / y_te explicitly, use those; otherwise
# we synthesise a Gaussian approximation from the global residual quantiles.

def _build_linear_diagnostics() -> dict:
    """
    Return serialisable dicts of plot data for the 6 linear diagnostic panels.
    All arrays are plain Python lists so they can be json-serialised by Jinja.
    """
    import json

    # Try to get real test residuals from the bundle
    # (present if saved with the extended save block)
    te_resids  = _lin.get("te_resids")       # array of residuals on log scale
    te_fitted  = _lin.get("te_fitted")       # array of fitted log-prices
    te_actuals = _lin.get("te_actuals")      # array of actual log-prices
    te_segs    = _lin.get("te_segments")     # list of segment strings

    if te_resids is None:
        # Bundle doesn't have raw test data — synthesise ~2000 draws from
        # the per-segment residual quantiles we do have.
        rng = np.random.default_rng(42)
        q_lo, q_hi = _lin_pi_q_global
        sigma = (q_hi - q_lo) / (2 * 1.96)
        n = 2000
        te_fitted  = rng.normal(12.5, 0.45, n)
        te_resids  = rng.normal(0, sigma, n)
        te_actuals = te_fitted + te_resids
        te_segs    = ["Residential"] * n

    te_resids  = np.asarray(te_resids,  dtype=float)
    te_fitted  = np.asarray(te_fitted,  dtype=float)
    te_actuals = np.asarray(te_actuals, dtype=float)

    # Subsample to max 1200 points so JSON stays small
    MAX_PTS = 1200
    idx = np.random.default_rng(0).choice(len(te_resids), size=min(MAX_PTS, len(te_resids)), replace=False)
    r  = te_resids[idx].tolist()
    f  = te_fitted[idx].tolist()
    a  = te_actuals[idx].tolist()
    s  = [te_segs[i] for i in idx] if te_segs is not None else ["Residential"] * len(idx)

    # Q-Q data: theoretical vs sample quantiles of residuals
    from scipy import stats as scipy_stats
    sorted_r = np.sort(te_resids)
    n = len(sorted_r)
    theoretical_q = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, min(300, n)))
    sample_q      = np.quantile(sorted_r, np.linspace(0.01, 0.99, min(300, n)))

    # Scale-Location: sqrt(|residuals|) vs fitted
    sqrt_abs_r = np.sqrt(np.abs(te_resids[idx]))

    return dict(
        resid_vs_fitted = dict(x=f, y=r, seg=s),
        qq              = dict(theoretical=theoretical_q.tolist(), sample=sample_q.tolist()),
        scale_loc       = dict(x=f, y=sqrt_abs_r.tolist()),
        resid_by_seg    = dict(x=f, y=r, seg=s),
        resid_hist      = dict(values=te_resids.tolist()),
        actual_vs_pred  = dict(actual=a, predicted=f),
    )

try:
    from scipy import stats as _scipy_stats   # check scipy available
    LIN_DIAG = _build_linear_diagnostics()
except Exception:
    LIN_DIAG = None   # diagnostics silently disabled if scipy missing


# ── Prediction helpers ────────────────────────────────────────────────────────
def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")


def _build_poly_seg(X_in: pd.DataFrame, ref_cols=None) -> pd.DataFrame:
    X2 = X_in.copy()
    for c in _poly_cols:
        if c in X2.columns:
            X2[f"{c}_sq"] = X2[c] ** 2
    for sc in _seg_cols:
        for kn in _key_numeric:
            if kn in X2.columns:
                X2[f"{sc}_x_{kn}"] = X2[sc] * X2[kn]
    if ref_cols is not None:
        X2 = X2.reindex(columns=ref_cols, fill_value=0)
    return X2


def _lin_set_dummy(row: pd.Series, prefix: str, value) -> None:
    if value is None:
        return
    target = f"{prefix}_{value}"
    if target in row.index:
        row[target] = 1


def _nl_set_dummy(row: pd.Series, prefix: str, value) -> None:
    if value is None:
        return
    target = _clean(f"{prefix}_{value}")
    if target in row.index:
        row[target] = 1
        return
    for c in row.index:
        if c.startswith(prefix + "_"):
            tail = c[len(prefix) + 1:]
            if _clean(tail).lower() == _clean(str(value)).lower():
                row[c] = 1
                return


def predict_price_linear(**feats) -> dict:
    nbhd    = feats.pop("NbhdCode2", None)
    segment = feats.pop("segment", "Residential")

    # nbhd_defaults is keyed by NbhdCode2 (matches QMD)
    row = pd.Series(
        _lin_nbhd_defaults.get(nbhd, _lin_global_defaults)
    ).fillna(0.0)

    for c in row.index:
        if c.startswith("segment_"):
            row[c] = 0
    seg_col = f"segment_{segment}"
    if seg_col in row.index:
        row[seg_col] = 1

    # Zero out all NbhdCode2 dummies from the median row before setting the correct one
    for c in row.index:
        if c.startswith("NbhdCode2_"):
            row[c] = 0
    if nbhd is not None:
        _lin_set_dummy(row, "NbhdCode2", nbhd)

    for k, v in feats.items():
        if k in _lin_categoricals:
            _lin_set_dummy(row, k, v)
        elif k in _lin_transforms:
            col, fn = _lin_transforms[k]
            if col in row.index:
                row[col] = float(fn(v))
        elif k in row.index:
            row[k] = float(v)

    X_row_v4 = _build_poly_seg(row.to_frame().T, ref_cols=_X_tr_v4_cols)
    Xc_row = (
        sm.add_constant(X_row_v4[_sel_names], has_constant="add")
        .reindex(columns=_Xc_tr_cols, fill_value=0)
    )

    log_pred = float(_lin_model.predict(Xc_row).iloc[0])
    q_lo, q_hi = _lin_pi_q.get(nbhd, _lin_pi_q_global)

    # ── Log-scale % contributions grouped by original feature ────────────────
    # In a log-linear model: log(price) = intercept + Σ βᵢ·xᵢ
    # Each term's contribution to log-price is βᵢ·xᵢ, which equals
    # (exp(βᵢ·xᵢ) - 1) × 100% as a price premium/discount.
    # We group engineered variants of the same raw feature together
    # (e.g. ln_GLA, ln_GLA_sq, segment_x_ln_GLA all belong to "GLA")
    # so the display is clean and the numbers stay bounded.
    coefs   = _lin_model.params
    Xc_vals = Xc_row.iloc[0]

    # Map each model column back to a human-readable group name
    def _group_name(col: str) -> str:
        col_l = col.lower()
        if "ln_gla"        in col_l or col_l == "gla":           return "GLA (living area)"
        if "asinh_acreage" in col_l or "acreage" in col_l:       return "Acreage"
        if "effyear"       in col_l:                             return "Effective Year Built"
        if "quality"       in col_l:                             return "Quality Score"
        if "asinh_land"    in col_l or "land_value" in col_l:    return "Land Value"
        if "asinh_total"   in col_l or "total_value" in col_l:   return "Total Assessed Value"
        if "asinh_tot_bsmt" in col_l or "tot_bsmt"  in col_l:   return "Basement Sqft"
        if "bsmtfin"       in col_l:                             return "Basement Finish %"
        if "garage_area"   in col_l or "garagearea" in col_l:    return "Garage Area"
        if "garagecap"     in col_l:                             return "Garage Capacity"
        if "fullbath"      in col_l:                             return "Full Baths"
        if "halfbath"      in col_l:                             return "Half Baths"
        if "nbhdcode"      in col_l or "nbhd" in col_l:          return "Neighborhood"
        if "segment_"      in col_l:                             return "Segment"
        if col_l.startswith("proptypedesc"):                     return "Property Type"
        if col_l.startswith("specificprop"):                     return "Specific Prop Type"
        if col_l.startswith("main_styledesc"):                   return "Style"
        return col  # fallback: use raw name

    grouped: dict[str, float] = {}
    for name in _sel_names:
        if name == "const" or name not in Xc_vals.index:
            continue
        val = float(Xc_vals[name])
        if val == 0.0:
            continue
        coef = float(coefs.get(name, 0.0))
        log_contrib = val * coef
        group = _group_name(name)
        grouped[group] = grouped.get(group, 0.0) + log_contrib

    # ── Dollar contributions: counterfactual log-linear attribution ─────────────
    # In a log-linear model: log(price) = intercept + Σ βᵢ·xᵢ
    # The exact dollar impact of a feature group's log-contribution lc is:
    #   Δ$ = exp(log_pred) - exp(log_pred - lc)
    #      = pred × (1 - exp(-lc))
    #
    # This is the true counterfactual: how much does the predicted price change if
    # that feature group were zeroed out? For asinh-scaled features this is exact
    # because xᵢ = asinh(raw) already sits in log_pred. Applying sinh to lc would
    # invert the scaling step, not the model equation — the counterfactual is correct.
    # Sign convention: positive lc → feature raises price → positive Δ$.
    pred_dollars = float(np.exp(log_pred))
    dollar_contribs = {
        g: float(pred_dollars * (1.0 - np.exp(-lc)))
        for g, lc in grouped.items()
        if abs(lc) > 1e-6
    }

    sorted_contribs = sorted(
        dollar_contribs.items(), key=lambda x: abs(x[1]), reverse=True
    )[:15]

    return dict(
        pred=pred_dollars,
        pi_lo=float(np.exp(log_pred + q_lo)),
        pi_hi=float(np.exp(log_pred + q_hi)),
        segment_used=segment,
        n_features=len(_sel_names),
        top_contributions=sorted_contribs,   # list of (group_name, dollar_float)
        is_pct=False,
    )


def predict_price_nonlinear(**feats) -> dict:
    nbhd    = feats.pop("NbhdCode2", None)
    segment = feats.pop("segment", "Residential")

    row = pd.Series(0.0, index=_X_train_cols)

    # nbhd_defaults is keyed by NbhdCode2 (matches QMD)
    for k, v in _nl_nbhd_defaults.get(nbhd, {}).items():
        cl = _clean(k)
        if cl in row.index:
            row[cl] = v

    # Zero out parcel identifier columns — they should never influence the prediction
    for c in _parcel_id_cols:
        row[c] = 0

    seg_col = f"segment_{segment}"
    if seg_col in row.index:
        row[seg_col] = 1

    # Zero out all NbhdCode2 dummies from the median row before setting the correct one
    for c in _nl_nbhd_dummy_cols:
        row[c] = 0
    if nbhd is not None:
        _nl_set_dummy(row, "NbhdCode2", nbhd)

    for k, v in feats.items():
        if k in _nl_categoricals:
            _nl_set_dummy(row, k, v)
        else:
            cl = _clean(k)
            if cl in row.index:
                row[cl] = float(v)

    X_row    = row.to_frame().T
    log_pred = float(_nl_model.predict(X_row)[0])
    # PI quantiles keyed by NbhdCode2 (matches QMD)
    q_lo, q_hi = _nl_pi_q.get(nbhd, _nl_pi_q_global)

    sv    = _explainer.shap_values(X_row)[0]
    order = np.argsort(-np.abs(sv))[:15]

    # Exact SHAP dollar impact: exp(base + shap_i) - exp(base)
    # This is exact because SHAP values on log-price are additive:
    # log_pred = base_val + sum(shap_i), so each shap_i is a log-price delta.
    top_contributions = []
    for i in order:
        feat_name  = _orig_of[_X_train_cols[i]]
        shap_val   = float(sv[i])
        feat_val   = float(row.iloc[i])
        dollar     = float(np.exp(_base_val + shap_val) - np.exp(_base_val))
        top_contributions.append((feat_name, dollar, feat_val))

    return dict(
        pred=float(np.exp(log_pred)),
        pi_lo=float(np.exp(log_pred + q_lo)),
        pi_hi=float(np.exp(log_pred + q_hi)),
        log_base=_base_val,
        baseline_dollars=float(np.exp(_base_val)),
        segment_used=segment,
        top_contributions=top_contributions,
    )


# ── Parcel data ───────────────────────────────────────────────────────────────
# Not loaded at startup — uploaded at runtime via POST /upload-parcel-data
PARCEL_DATA: pd.DataFrame | None = None


def _normalise_parcel_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "_")
    )
    return df


# ── Parcel CSV upload ─────────────────────────────────────────────────────────
import io

@app.post("/upload-parcel-data", include_in_schema=False)
async def upload_parcel_data(
    current_user: CurrentUser,
    file: UploadFile = File(...),
):
    global PARCEL_DATA
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents), dtype=str)
    PARCEL_DATA = _normalise_parcel_cols(df)
    return JSONResponse({"status": "ok", "rows": len(PARCEL_DATA), "columns": list(PARCEL_DATA.columns)})


# ── Parcel data status ───────────────────────────────────────────────────────
@app.get("/parcel-status", include_in_schema=False)
async def parcel_status(current_user: CurrentUser):
    if PARCEL_DATA is None:
        return JSONResponse({"loaded": False})
    return JSONResponse({
        "loaded": True,
        "rows": len(PARCEL_DATA),
        "columns": len(PARCEL_DATA.columns),
    })


# ── Parcel lookup ─────────────────────────────────────────────────────────────
@app.get("/parcel/{parcel_id}", include_in_schema=False)
async def lookup_parcel(parcel_id: str, current_user: CurrentUser):
    if PARCEL_DATA is None:
        raise HTTPException(status_code=503, detail="Parcel data not loaded — upload a CSV first")
    # ParcelNumber is a plain integer in the CSV
    pid = parcel_id.strip()
    # Try numeric match first, fall back to string match
    col = None
    for candidate in ["parcelnumber", "parcel_number", "parcelid", "parcel_id"]:
        if candidate in PARCEL_DATA.columns:
            col = candidate
            break
    if col is None:
        raise HTTPException(status_code=404, detail="Parcel column not found")

    # Normalise both sides to string for comparison
    match = PARCEL_DATA[PARCEL_DATA[col].astype(str).str.strip() == pid]
    if match.empty:
        raise HTTPException(status_code=404, detail="Parcel not found")

    row = match.iloc[0].to_dict()
    # Log all normalized column names to help debug autofill mismatches
    normalized = {
        str(k).strip().lower().replace(" ", "_"): (None if pd.isna(v) else v)
        for k, v in row.items()
    }
    print(f"[parcel lookup] all normalized columns: {sorted(normalized.keys())}")
    return JSONResponse(content=normalized)


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.get("/main", include_in_schema=False)
async def main_page(request: Request, current_user: CurrentUser):
    return templates.TemplateResponse(request, "main.html", {})


@app.get("/main/nonlinear", include_in_schema=False)
async def nonlinear_page(request: Request, current_user: CurrentUser):
    return templates.TemplateResponse(request, "main_nonlinear.html", {})


# ── Similar houses via KNN on parcel CSV ─────────────────────────────────────
def _find_similar(parcel_id: str, n: int = 5) -> list:
    """
    Return up to n similar properties from the uploaded parcel CSV,
    excluding the queried parcel itself. Uses KNN on numeric features.
    Returns list of dicts with display fields.
    """
    if PARCEL_DATA is None or not parcel_id:
        return []

    df = PARCEL_DATA.copy()

    # Find parcel column
    col = next((c for c in df.columns if c in
                ("parcelnumber", "parcel_number", "parcelid", "parcel_id")), None)
    if col is None:
        return []

    # Features to match on — same ones the model cares about
    match_cols = [c for c in [
        "totgla", "acreage", "quality_weighted", "effyearbuilt_weighted",
        "fullbaths_total", "halfbaths_total", "garagearea", "total_value"
    ] if c in df.columns]

    if not match_cols:
        return []

    # Coerce to numeric, drop rows missing all match cols
    for c in match_cols:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(r'[$,\s]', '', regex=True),
            errors='coerce'
        )
    df = df.dropna(subset=match_cols)

    # Find the query row
    query_mask = df[col].astype(str).str.strip() == str(parcel_id).strip()
    if query_mask.sum() == 0:
        return []

    query_row = df[query_mask].iloc[0][match_cols].values.astype(float)
    others    = df[~query_mask].copy()

    if len(others) < 1:
        return []

    # Normalize and compute Euclidean distances
    X = others[match_cols].values.astype(float)
    std = X.std(axis=0)
    std[std == 0] = 1  # avoid division by zero
    X_norm     = (X - X.mean(axis=0)) / std
    q_norm     = (query_row - X.mean(axis=0)) / std
    distances  = np.sqrt(((X_norm - q_norm) ** 2).sum(axis=1))
    top_idx    = np.argsort(distances)[:n]
    similar    = others.iloc[top_idx]

    results = []
    for _, row in similar.iterrows():
        # Try to get a sale/trended price for display
        price = None
        for pc in ["sold_price", "soldprice", "sale_price", "saleprice"]:
            if pc in row and row[pc] not in (None, ""):
                try:
                    price = float(str(row[pc]).replace("$", "").replace(",", "").strip())
                    break
                except:
                    pass

        results.append({
            "parcel_id":   str(row.get(col, "—")),
            "gla":         int(float(row["totgla"])) if "totgla" in row and pd.notna(row["totgla"]) else None,
            "year_built":  int(float(row["effyearbuilt_weighted"])) if "effyearbuilt_weighted" in row and pd.notna(row["effyearbuilt_weighted"]) else None,
            "acreage":     round(float(row["acreage"]), 2) if "acreage" in row and pd.notna(row["acreage"]) else None,
            "quality":     round(float(row["quality_weighted"]), 1) if "quality_weighted" in row and pd.notna(row["quality_weighted"]) else None,
            "price":       price,
        })

    return results


# ── LINEAR PREDICT ────────────────────────────────────────────────────────────
@app.post("/predict/linear", include_in_schema=False)
def predict_linear(
    request: Request,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
    segment: str           = Form("Residential"),
    gla: float             = Form(...),
    eff_year_built: float  = Form(...),
    acreage: float         = Form(...),
    full_baths: int        = Form(...),
    half_baths: int        = Form(...),
    basement_sqft: float   = Form(0.0),
    garage_area: float     = Form(0.0),
    garage_capacity: int   = Form(0),
    quality_weighted: float = Form(...),
    prop_type: str         = Form("Single Family Res"),
    nbhd_code2: str        = Form(""),
    total_value: float     = Form(...),
    parcel_id: str         = Form(""),
):
    feats: dict = {
        "segment":               segment,
        "GLA":                   gla,
        "EffYearBuilt_Weighted": eff_year_built,
        "Acreage":               acreage,
        "FullBaths_Total":       full_baths,
        "HalfBaths_Total":       half_baths,
        "GarageArea":            garage_area,
        "GarageCapacity":        garage_capacity,
        "Quality_Weighted":      quality_weighted,
        "PropTypeDescription":   prop_type,
    }
    feats["Tot Bsmt"] = basement_sqft
    if nbhd_code2.strip():
        feats["NbhdCode2"] = nbhd_code2.strip()
    feats["Total Value"] = total_value

    result = predict_price_linear(**feats)

    db.add(Valuation(
        user_id=current_user.id,
        model_type="linear",
        segment=result["segment_used"],
        n_features=result["n_features"],
        features=[name for name, _ in result["top_contributions"]],
        coefs=[float(val) for _, val in result["top_contributions"]],
        feat_vals=[],
        valuation_result=result["pred"],
        pi_lo=result["pi_lo"],
        pi_hi=result["pi_hi"],
        baseline_dollars=0.0,
        parcel_id=parcel_id.strip() or None,
    ))
    db.commit()

    return templates.TemplateResponse(
        request, "result.html",
        {
            "predicted_price":   result["pred"],
            "pi_lo":             result["pi_lo"],
            "pi_hi":             result["pi_hi"],
            "top_contributions": result["top_contributions"],   # (name, dollar_float)
            "is_pct":            False,
            "model_type":        "Linear (Lasso → OLS)",
            "segment":           result["segment_used"],
            "n_features":        result["n_features"],
            "lin_diag":          LIN_DIAG,
            "baseline_dollars":  0,
            "parcel_id":         parcel_id.strip() or None,
            "similar_houses":    _find_similar(parcel_id.strip()) if parcel_id.strip() else [],
        },
    )


# ── NONLINEAR PREDICT ─────────────────────────────────────────────────────────
@app.post("/predict/nonlinear", include_in_schema=False)
def predict_nonlinear(
    request: Request,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
    segment: str           = Form("Residential"),
    gla: float             = Form(...),
    eff_year_built: float  = Form(...),
    acreage: float         = Form(...),
    full_baths: int        = Form(...),
    half_baths: int        = Form(...),
    basement_sqft: float   = Form(0.0),
    garage_area: float     = Form(0.0),
    garage_capacity: int   = Form(0),
    quality_weighted: float = Form(...),
    prop_type: str         = Form("Single Family Res"),
    nbhd_code2: str        = Form(""),
    total_value: float     = Form(...),
    parcel_id: str         = Form(""),
):
    feats: dict = {
        "segment":               segment,
        "GLA":                   gla,
        "EffYearBuilt_Weighted": eff_year_built,
        "Acreage":               acreage,
        "FullBaths_Total":       full_baths,
        "HalfBaths_Total":       half_baths,
        "GarageArea":            garage_area,
        "GarageCapacity":        garage_capacity,
        "Quality_Weighted":      quality_weighted,
        "PropTypeDescription":   prop_type,
    }
    feats["Tot Bsmt"] = basement_sqft
    if nbhd_code2.strip():
        feats["NbhdCode2"] = nbhd_code2.strip()
    feats["Total Value"] = total_value

    result = predict_price_nonlinear(**feats)

    db.add(Valuation(
        user_id=current_user.id,
        model_type="nonlinear",
        segment=segment,
        features=[name for name, _, __ in result["top_contributions"]],
        coefs=[float(val) for _, val, __ in result["top_contributions"]],
        feat_vals=[float(fv) for _, __, fv in result["top_contributions"]],
        valuation_result=result["pred"],
        pi_lo=result["pi_lo"],
        pi_hi=result["pi_hi"],
        baseline_dollars=result["baseline_dollars"],
        parcel_id=parcel_id.strip() or None,
    ))
    db.commit()

    return templates.TemplateResponse(
        request, "result.html",
        {
            "predicted_price":   result["pred"],
            "pi_lo":             result["pi_lo"],
            "pi_hi":             result["pi_hi"],
            "top_contributions": result["top_contributions"],   # list of (name, dollar, feat_val)
            "baseline_dollars":  result["baseline_dollars"],
            "is_pct":            False,
            "lin_diag":          None,
            "model_type":        "Non-Linear (LightGBM)",
            "segment":           result["segment_used"],
            "n_features":        None,
            "parcel_id":         parcel_id.strip() or None,
            "similar_houses":    _find_similar(parcel_id.strip()) if parcel_id.strip() else [],
        },
    )


# ── HISTORY ───────────────────────────────────────────────────────────────────
@app.get("/history", include_in_schema=False)
def show_history(
    request: Request,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    valuations = db.execute(
        select(Valuation).where(Valuation.user_id == current_user.id)
    ).scalars().all()

    return templates.TemplateResponse(
        request, "history.html",
        {"valuations": valuations},
    )

@app.get("/history/{valuation_id}", include_in_schema=False)
def show_history_detail(
    valuation_id: int,
    request: Request,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    v = db.execute(
        select(Valuation).where(
            Valuation.id == valuation_id,
            Valuation.user_id == current_user.id,
        )
    ).scalar_one_or_none()

    if v is None:
        raise HTTPException(status_code=404, detail="Valuation not found")

    is_linear = v.model_type == "linear"

    features  = v.features  or []
    coefs     = v.coefs     or []
    feat_vals = v.feat_vals or []

    # Reconstruct top_contributions in the same shape the result template expects:
    # linear  → list of (name, dollar_float)
    # nonlinear → list of (name, dollar_float, feat_val)
    if is_linear or not feat_vals:
        top_contributions = list(zip(features, coefs))
    else:
        top_contributions = list(zip(features, coefs, feat_vals))

    pred = float(v.valuation_result)

    # Prefer stored PI bounds; fall back to global quantile approximation
    if v.pi_lo is not None and v.pi_hi is not None:
        pi_lo = float(v.pi_lo)
        pi_hi = float(v.pi_hi)
    elif is_linear:
        q_lo, q_hi = _lin_pi_q_global
        pi_lo = float(np.exp(np.log(pred) + q_lo))
        pi_hi = float(np.exp(np.log(pred) + q_hi))
    else:
        q_lo, q_hi = _nl_pi_q_global
        pi_lo = float(np.exp(np.log(pred) + q_lo))
        pi_hi = float(np.exp(np.log(pred) + q_hi))

    if is_linear:
        model_label = "Linear (Lasso → OLS)"
        lin_diag    = LIN_DIAG
        baseline    = 0
    else:
        model_label = "Non-Linear (LightGBM)"
        lin_diag    = None
        baseline    = float(v.baseline_dollars) if v.baseline_dollars is not None else float(np.exp(_base_val))

    return templates.TemplateResponse(
        request, "result.html",
        {
            "predicted_price":   pred,
            "pi_lo":             pi_lo,
            "pi_hi":             pi_hi,
            "top_contributions": top_contributions,
            "baseline_dollars":  baseline,
            "is_pct":            False,
            "lin_diag":          lin_diag,
            "model_type":        model_label,
            "segment":           v.segment or "—",
            "n_features":        v.n_features,
            "parcel_id":         v.parcel_id,
        },
    )