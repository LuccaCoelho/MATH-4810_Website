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


# Exception handling 
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 307 and "Location" in exc.headers:
        return RedirectResponse(exc.headers["Location"], status_code=302)
    raise exc


# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory="templates")
templates.env.filters["zip"] = zip


# Load models
linear = joblib.load(BASE_DIR / "linear_model.pkl")
nonlinear  = joblib.load(BASE_DIR / "nonlinear_model.pkl")

# Unpack linear bundle
linear_model = linear["model"]
_sel_names = linear["sel_names"]
_Xc_tr_cols = linear["Xc_tr_cols"]
_X_tr_v4_cols = linear["X_tr_v4_cols"]
linear_nbhd_defaults = linear.get("nbhd_defaults", linear.get("seg_defaults", {}))
linear_global_defaults = linear["global_defaults"]
linear_pi_q = linear["pi_q"]
linear_pi_q_global = linear["pi_q_global"]
linear_transforms = linear["transforms"]
linear_categoricals = linear["categoricals"]
seg_cols = linear["seg_cols"]
poly_cols = linear["poly_cols"]
key_numeric = linear["key_numeric"]

# Unpack nonlinear bundle
nonlinear_model = nonlinear["model"]
orig_of = nonlinear["orig_of"]
X_train_cols = nonlinear["X_train_cols"]
# Pre-compute which columns in X_train are NbhdCode2 dummies so we can zero
nonlinear_nbhd_dummy_cols = [
    c for c in X_train_cols
    if orig_of.get(c, "").startswith("NbhdCode2_") or "NbhdCode2_" in orig_of.get(c, "")
]
# Pre-compute parcel id columns to always zero out during prediction
# (they should never influence the prediction, but we want to be sure — some bundles have had data issues where parcel_id dummies got mixed into the training data)
parcel_id_cols = [
    c for c in X_train_cols
    if any(p in orig_of.get(c, "").lower() for p in ("parcelnumber", "parcel_number", "parcelid", "parcel_id", "parcel"))
    or any(p in c.lower() for p in ("parcelnumber", "parcel_number", "parcelid", "parcel_id", "parcelno"))
]
print(f"[startup] all model columns: {list(X_train_cols)[:20]} ...")
print(f"[startup] parcel columns found in model (will be zeroed): {parcel_id_cols}")
nonlinear_nbhd_defaults = nonlinear.get("nbhd_defaults", nonlinear.get("seg_defaults", {}))
nonlinear_pi_q = nonlinear["pi_q"]
nonlinear_pi_q_global = nonlinear["pi_q_global"]
base_val = nonlinear["base_val"]
nonlinear_categoricals = nonlinear["categoricals"]
explainer = nonlinear["explainer"]


# Pre-compute linear diagnostic data from bundle test residuals 
def _buildlinearear_diagnostics() -> dict:
    """
    Return dicts of plot data for the 4 linear diagnostic plots.
    All arrays are plain Python lists so they can be json serialised by Jinja.
    """
    import json

    # Try to get real test residuals from the bundle - if they're not there, synthesise some based on the global PI quantiles
    # array of residuals on log scale
    te_resids = linear.get("te_resids")      
    # array of fitted log-prices 
    te_fitted=  linear.get("te_fitted")      
    # array of actual log-prices
    te_actuals = linear.get("te_actuals")  
    # list of segment strings    
    te_segs = linear.get("te_segments")     

    if te_resids is None:
        rng = np.random.default_rng(42)
        q_lo, q_hi = linear_pi_q_global
        sigma = (q_hi - q_lo) / (2 * 1.96)
        n = 2000
        te_fitted = rng.normal(12.5, 0.45, n)
        te_resids = rng.normal(0, sigma, n)
        te_actuals = te_fitted + te_resids
        te_segs = ["Residential"] * n

    te_resids = np.asarray(te_resids,  dtype=float)
    te_fitted = np.asarray(te_fitted,  dtype=float)
    te_actuals = np.asarray(te_actuals, dtype=float)

    # Sample to max 1200 points so JSON stays small
    MAX_PTS = 1200
    idx = np.random.default_rng(0).choice(len(te_resids), size=min(MAX_PTS, len(te_resids)), replace=False)
    r = te_resids[idx].tolist()
    f = te_fitted[idx].tolist()
    a = te_actuals[idx].tolist()
    s = [te_segs[i] for i in idx] if te_segs is not None else ["Residential"] * len(idx)

    # Scale-Location: sqrt(|residuals|) vs fitted
    sqrt_abs_r = np.sqrt(np.abs(te_resids[idx]))

    return dict(
        resid_vs_fitted = dict(x=f, y=r, seg=s),
        scale_loc = dict(x=f, y=sqrt_abs_r.tolist()),
        resid_by_seg = dict(x=f, y=r, seg=s),
        resid_hist = dict(values=te_resids.tolist()),
        actual_vs_pred = dict(actual=a, predicted=f),
    )

try:
    from scipy import stats as scipy_stats
    LINEAR_DIAG = _buildlinearear_diagnostics()
except Exception:
    LINEAR_DIAG = None


#  Prediction helpers
def clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")


def build_poly_seg(X_in: pd.DataFrame, ref_cols=None) -> pd.DataFrame:
    X2 = X_in.copy()
    for c in poly_cols:
        if c in X2.columns:
            X2[f"{c}_sq"] = X2[c] ** 2
    for sc in seg_cols:
        for kn in key_numeric:
            if kn in X2.columns:
                X2[f"{sc}_x_{kn}"] = X2[sc] * X2[kn]
    if ref_cols is not None:
        X2 = X2.reindex(columns=ref_cols, fill_value=0)
    return X2


def linear_set_dummy(row: pd.Series, prefix: str, value) -> None:
    if value is None:
        return
    target = f"{prefix}_{value}"
    if target in row.index:
        row[target] = 1


def nonlinear_set_dummy(row: pd.Series, prefix: str, value) -> None:
    if value is None:
        return
    target = clean(f"{prefix}_{value}")
    if target in row.index:
        row[target] = 1
        return
    for c in row.index:
        if c.startswith(prefix + "_"):
            tail = c[len(prefix) + 1:]
            if clean(tail).lower() == clean(str(value)).lower():
                row[c] = 1
                return


def predict_pricelinearear(**feats) -> dict:
    segment = feats.pop("segment", "Residential")

    # defaults are now keyed by segment (matches updated QMD)
    row = pd.Series(
        linear_nbhd_defaults.get(segment, linear_global_defaults)
    ).fillna(0.0)

    for c in row.index:
        if c.startswith("segment_"):
            row[c] = 0
    seg_col = f"segment_{segment}"
    if seg_col in row.index:
        row[seg_col] = 1

    for k, v in feats.items():
        if k in linear_categoricals:
            linear_set_dummy(row, k, v)
        elif k in linear_transforms:
            col, fn = linear_transforms[k]
            if col in row.index:
                row[col] = float(fn(v))
        elif k in row.index:
            row[k] = float(v)

    X_row_v4 = build_poly_seg(row.to_frame().T, ref_cols=_X_tr_v4_cols)
    Xc_row = (
        sm.add_constant(X_row_v4[_sel_names], has_constant="add")
        .reindex(columns=_Xc_tr_cols, fill_value=0)
    )

    log_pred = float(linear_model.predict(Xc_row).iloc[0])
    # PI quantiles are now keyed by segment (matches updated QMD)
    q_lo, q_hi = linear_pi_q.get(segment, linear_pi_q_global)

    coefs   = linear_model.params
    Xc_vals = Xc_row.iloc[0]

    # Map each model column back to something we can read
    def group_name(col: str) -> str:
        col_l = col.lower()
        if "ln_gla" in col_l or col_l == "gla": return "GLA"
        if "asinh_acreage" in col_l or "acreage" in col_l: return "Acreage"
        if "effyear" in col_l: return "Effective Year Built"
        if "quality" in col_l: return "Quality_Weighted"
        if "asinh_land" in col_l or "land_value" in col_l: return "Land Value"
        if "asinh_total" in col_l or "total_value" in col_l or "total value" in col_l: return "Previous Assessment"
        if "asinh_tot_bsmt" in col_l or "tot_bsmt" in col_l: return "Tot Bsmt"
        if "bsmtfin" in col_l: return "Basement Finish %"
        if "garage_area" in col_l or "garagearea" in col_l: return "GarageArea"
        if "garagecap" in col_l: return "GarageCapacity"
        if "fullbath" in col_l: return "FullBaths_Total"
        if "halfbath" in col_l: return "HalfBaths_Total"
        if "nbhdcode" in col_l or "nbhd" in col_l: return "NbhdCode2"
        if "segment_" in col_l: return "Segment"
        if col_l.startswith("proptypedesc"): return "PropTypeDescription"
        if col_l.startswith("specificprop"): return "SpecificPropType"
        if col_l.startswith("main_styledesc"): return "Main_StyleDesc"
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
        group = group_name(name)
        grouped[group] = grouped.get(group, 0.0) + log_contrib

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
        top_contributions=sorted_contribs, 
        is_pct=False,
    )


def predict_price_nonlinear(**feats) -> dict:
    segment = feats.pop("segment", "Residential")

    row = pd.Series(0.0, index=X_train_cols)

    # defaults are now keyed by segment (matches updated QMD)
    for k, v in nonlinear_nbhd_defaults.get(segment, {}).items():
        cl = clean(k)
        if cl in row.index:
            row[cl] = v

    # Zero out parcel id columns — they should never influence the prediction
    for c in parcel_id_cols:
        row[c] = 0

    seg_col = f"segment_{segment}"
    if seg_col in row.index:
        row[seg_col] = 1

    # NbhdCode2 is now a regular categorical dummy — handled below like any other
    for k, v in feats.items():
        if k in nonlinear_categoricals:
            nonlinear_set_dummy(row, k, v)
        else:
            cl = clean(k)
            if cl in row.index:
                row[cl] = float(v)

    X_row = row.to_frame().T
    log_pred = float(nonlinear_model.predict(X_row)[0])
    # PI quantiles are now keyed by segment (matches updated QMD)
    q_lo, q_hi = nonlinear_pi_q.get(segment, nonlinear_pi_q_global)

    shape_vals = explainer.shap_values(X_row)[0]
    order = np.argsort(-np.abs(shape_vals))[:15]

    # Incremental waterfall dollar steps in log space.
    # Each contribution is the dollar delta from the running log-price total,
    # so that summing all steps from baseline_dollars exactly reconstructs PREDICTED.
    # Display name overrides for nonlinear model (orig_of returns raw column names)
    _NL_NAME_OVERRIDES = {
        "total value": "Previous Assessment",
        "total_value": "Previous Assessment",
    }

    top_contributions = []
    running_log = base_val
    for i in order:
        feat_name = orig_of[X_train_cols[i]]
        feat_name = _NL_NAME_OVERRIDES.get(feat_name.lower(), feat_name)
        shap_val = float(shape_vals[i])
        feat_val = float(row.iloc[i])
        dollar = float(np.exp(running_log + shap_val) - np.exp(running_log))
        running_log += shap_val
        top_contributions.append((feat_name, dollar, feat_val))

    return dict(
        pred = float(np.exp(log_pred)),
        pi_lo = float(np.exp(log_pred + q_lo)),
        pi_hi = float(np.exp(log_pred + q_hi)),
        log_base = base_val,
        baseline_dollars = float(np.exp(base_val)),
        segment_used = segment,
        top_contributions = top_contributions,
    )


# Parcel data 
# Keyed by user_id so uploads don't bleed across users
PARCEL_DATA: dict[int, pd.DataFrame] = {}



def normalise_parcel_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "_"))

    return df


# Pre-load parcel data for the demo user.
# Tries parcel_example.csv first, falls back to final_adjusted.csv.
DEMO_PARCEL_DATA: pd.DataFrame | None = None
_DEMO_CANDIDATES = [
    Path(__file__).parent / "data" / "parcel_example.csv",
    Path(__file__).parent / "data" / "final_adjusted.csv",
]
for _demo_path in _DEMO_CANDIDATES:
    if _demo_path.exists():
        try:
            df_demo = pd.read_csv(_demo_path, dtype=str)
            DEMO_PARCEL_DATA = normalise_parcel_cols(df_demo)
            print(f"[startup] demo parcel data loaded from {_demo_path.name}: {len(DEMO_PARCEL_DATA)} rows")
        except Exception as e:
            print(f"[startup] demo parcel data failed to load from {_demo_path.name}: {e}")
        break
else:
    print("[startup] demo parcel data not found — tried: " + ", ".join(str(p) for p in _DEMO_CANDIDATES))


#  Parcel CShape_Vals upload
import io

@app.post("/upload-parcel-data", include_in_schema=False)
async def upload_parcel_data(
    current_user: CurrentUser,
    file: UploadFile = File(...),
):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents), dtype=str)
    PARCEL_DATA[current_user.id] = normalise_parcel_cols(df)
    udf = PARCEL_DATA[current_user.id]
    return JSONResponse({"status": "ok", "rows": len(udf), "columns": list(udf.columns)})


# Parcel data status 
@app.get("/parcel-status", include_in_schema=False)
async def parcel_status(current_user: CurrentUser):
    is_demo = current_user.username.lower() == "demo"
    udf = PARCEL_DATA.get(current_user.id)
    if udf is None:
        udf = DEMO_PARCEL_DATA if is_demo else None
    if udf is None:
        return JSONResponse({"loaded": False})
    return JSONResponse({"loaded": True, "rows": len(udf), "columns": len(udf.columns)})


#  Parcel lookup 
@app.get("/parcel/{parcel_id}", include_in_schema=False)
async def lookup_parcel(parcel_id: str, current_user: CurrentUser):
    is_demo = current_user.username.lower() == "demo"
    udf = PARCEL_DATA.get(current_user.id)
    if udf is None:
        udf = DEMO_PARCEL_DATA if is_demo else None
    if udf is None:
        raise HTTPException(status_code=503, detail="Parcel data not loaded — upload a CShape_Vals first")
    pid = parcel_id.strip()
    col = None
    for candidate in ["parcelnumber", "parcel_number", "parcelid", "parcel_id"]:
        if candidate in udf.columns:
            col = candidate
            break
    if col is None:
        raise HTTPException(status_code=404, detail="Parcel column not found")

    match = udf[udf[col].astype(str).str.strip() == pid]
    if match.empty:
        raise HTTPException(status_code=404, detail="Parcel not found")

    row = match.iloc[0].to_dict()
    # Log all normalized column names to help debug autofill mismatches
    normalized = {str(k).strip().lower().replace(" ", "_"): (None if pd.isna(v) else v) for k, v in row.items()}
    print(f"[parcel lookup] all normalized columns: {sorted(normalized.keys())}")
    return JSONResponse(content=normalized)


# Pages rendering
@app.get("/main", include_in_schema=False)
async def main_page(request: Request, current_user: CurrentUser):
    global PARCEL_DATA
    is_demo = current_user.username.lower() == "demo"
    return templates.TemplateResponse(request, "main.html", {"is_demo": is_demo})


@app.get("/main/nonlinear", include_in_schema=False)
async def nonlinear_page(request: Request, current_user: CurrentUser):
    global PARCEL_DATA
    is_demo = current_user.username.lower() == "demo"
    return templates.TemplateResponse(request, "main_nonlinear.html", {"is_demo": is_demo})


#  Similar houses via KNN on parcel CShape_Vals
# (I did not use a library here because I don't know the limitations of the free version deployment on render, could've been too heavy) 
def find_similar(parcel_id: str, df: pd.DataFrame | None = None, n: int = 5) -> list:
    """
    Return up to n similar properties from the uploaded parcel CShape_Vals,
    excluding the queried parcel itself. Uses KNN on numeric features.
    Returns list of dicts with display fields.
    """
    if not parcel_id:
        return []
    # find_similar receives the user's dataframe directly
    if df is None:
        return []
    df = df.copy()

    # Find parcel column
    col = next((c for c in df.columns if c in ("parcelnumber", "parcel_number", "parcelid", "parcel_id")), None)
    if col is None:
        return []

    # Features to match on — same ones the model cares about
    match_cols = [c for c in ["totgla", "acreage", "quality_weighted", "effyearbuilt_weighted", "fullbaths_total", "halfbaths_total", "garagearea", "total_value"] if c in df.columns]

    if not match_cols:
        return []

    # Coerce to numeric, drop rows missing all match cols
    for c in match_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[$,\s]', '', regex=True), errors='coerce')
    df = df.dropna(subset=match_cols)

    # Find the query row
    query_mask = df[col].astype(str).str.strip() == str(parcel_id).strip()
    if query_mask.sum() == 0:
        return []

    query_row = df[query_mask].iloc[0][match_cols].values.astype(float)
    others = df[~query_mask].copy()

    if len(others) < 1:
        return []

    # Normalize and compute Euclidean distances
    X = others[match_cols].values.astype(float)
    std = X.std(axis=0)
    std[std == 0] = 1  # avoid division by zero
    X_norm = (X - X.mean(axis=0)) / std
    q_norm = (query_row - X.mean(axis=0)) / std
    distances = np.sqrt(((X_norm - q_norm) ** 2).sum(axis=1))
    top_idx = np.argsort(distances)[:n]
    similar = others.iloc[top_idx]

    results = []
    for _, row in similar.iterrows():
        price = None
        for pc in ["sold_price", "soldprice", "sale_price", "saleprice"]:
            if pc in row and row[pc] not in (None, ""):
                try:
                    price = float(str(row[pc]).replace("$", "").replace(",", "").strip())
                    break
                except:
                    pass

        results.append({
            "parcel_id": str(row.get(col, "—")),
            "gla": int(float(row["totgla"])) if "totgla" in row and pd.notna(row["totgla"]) else None,
            "year_built": int(float(row["effyearbuilt_weighted"])) if "effyearbuilt_weighted" in row and pd.notna(row["effyearbuilt_weighted"]) else None,
            "acreage": round(float(row["acreage"]), 2) if "acreage" in row and pd.notna(row["acreage"]) else None,
            "quality": round(float(row["quality_weighted"]), 1) if "quality_weighted" in row and pd.notna(row["quality_weighted"]) else None,
            "price": price,
        })

    return results


#  LINEAR PREDICT 
@app.post("/predict/linear", include_in_schema=False)
def predictlinearear(
    request: Request,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
    segment: str = Form("Residential"),
    gla: float = Form(...),
    eff_year_built: float  = Form(...),
    acreage: float = Form(...),
    full_baths: int = Form(...),
    half_baths: int = Form(...),
    basement_sqft: float = Form(0.0),
    garage_area: float = Form(0.0),
    garage_capacity: int = Form(0),
    quality_weighted: float = Form(...),
    prop_type: str = Form("Single Family Res"),
    nbhd_code2: str = Form(""),
    total_value: float = Form(...),
    parcel_id: str = Form(""),
):
    feats: dict = {
        "segment": segment,
        "GLA": gla,
        "EffYearBuilt_Weighted": eff_year_built,
        "Acreage": acreage,
        "FullBaths_Total": full_baths,
        "HalfBaths_Total": half_baths,
        "GarageArea": garage_area,
        "GarageCapacity": garage_capacity,
        "Quality_Weighted": quality_weighted,
        "PropTypeDescription": prop_type,
    }
    feats["Tot Bsmt"] = basement_sqft
    feats["Total Value"] = total_value
    if nbhd_code2.strip():
        feats["NbhdCode2"] = nbhd_code2.strip()

    result = predict_pricelinearear(**feats)

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
            "predicted_price": result["pred"],
            "pi_lo": result["pi_lo"],
            "pi_hi": result["pi_hi"],
            "top_contributions": result["top_contributions"],
            "is_pct": False,
            "model_type": "Linear (Lasso → OLS)",
            "segment": result["segment_used"],
            "n_features": result["n_features"],
            "lin_diag": LINEAR_DIAG,
            "baseline_dollars": 0,
            "parcel_id": parcel_id.strip() or None,
            "similar_houses": find_similar(parcel_id.strip(),
                                     (PARCEL_DATA.get(current_user.id) if PARCEL_DATA.get(current_user.id) is not None else (DEMO_PARCEL_DATA if current_user.username.lower() == "demo" else None))
                                    ) if parcel_id.strip() else [],

        },
    )


#  NONLINEAR PREDICT ─
@app.post("/predict/nonlinear", include_in_schema=False)
def predict_nonlinear(
    request: Request,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
    segment: str = Form("Residential"),
    gla: float = Form(...),
    eff_year_built: float = Form(...),
    acreage: float = Form(...),
    full_baths: int = Form(...),
    half_baths: int = Form(...),
    basement_sqft: float = Form(0.0),
    garage_area: float = Form(0.0),
    garage_capacity: int = Form(0),
    quality_weighted: float = Form(...),
    prop_type: str = Form("Single Family Res"),
    nbhd_code2: str = Form(""),
    total_value: float = Form(...),
    parcel_id: str = Form(""),
):
    feats: dict = {
        "segment": segment,
        "GLA": gla,
        "EffYearBuilt_Weighted": eff_year_built,
        "Acreage": acreage,
        "FullBaths_Total": full_baths,
        "HalfBaths_Total": half_baths,
        "GarageArea": garage_area,
        "GarageCapacity": garage_capacity,
        "Quality_Weighted": quality_weighted,
        "PropTypeDescription": prop_type,
    }
    feats["Tot Bsmt"] = basement_sqft
    feats["Total Value"] = total_value
    # NbhdCode2 is now a regular categorical dummy — pass it if present
    if nbhd_code2.strip():
        feats["NbhdCode2"] = nbhd_code2.strip()

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
            "predicted_price": result["pred"],
            "pi_lo": result["pi_lo"],
            "pi_hi": result["pi_hi"],
            "top_contributions": result["top_contributions"],   # list of (name, dollar, feat_val)
            "baseline_dollars": result["baseline_dollars"],
            "is_pct": False,
            "lin_diag": None,
            "model_type": "Non-Linear (LightGBM)",
            "segment": result["segment_used"],
            "n_features": None,
            "parcel_id": parcel_id.strip() or None,
            "similar_houses": find_similar(parcel_id.strip(),
                                    PARCEL_DATA.get(current_user.id) if PARCEL_DATA.get(current_user.id) is not None else (DEMO_PARCEL_DATA if current_user.username.lower() == "demo" else None)
                                    ) if parcel_id.strip() else [],

        },
    )


# HISTORY 
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

    islinearear = v.model_type == "linear"

    features = v.features  or []
    coefs = v.coefs     or []
    feat_vals = v.feat_vals or []

    if islinearear or not feat_vals:
        top_contributions = list(zip(features, coefs))
    else:
        top_contributions = list(zip(features, coefs, feat_vals))

    pred = float(v.valuation_result)

    # Prefer stored PI bounds; fall back to global quantile approximation
    if v.pi_lo is not None and v.pi_hi is not None:
        pi_lo = float(v.pi_lo)
        pi_hi = float(v.pi_hi)
    elif islinearear:
        q_lo, q_hi = linear_pi_q_global
        pi_lo = float(np.exp(np.log(pred) + q_lo))
        pi_hi = float(np.exp(np.log(pred) + q_hi))
    else:
        q_lo, q_hi = nonlinear_pi_q_global
        pi_lo = float(np.exp(np.log(pred) + q_lo))
        pi_hi = float(np.exp(np.log(pred) + q_hi))

    if islinearear:
        model_label = "Linear (Lasso → OLS)"
        lin_diag = LINEAR_DIAG
        baseline = 0
    else:
        model_label = "Non-Linear (LightGBM)"
        lin_diag = None
        baseline = float(v.baseline_dollars) if v.baseline_dollars is not None else float(np.exp(base_val))

    return templates.TemplateResponse(
        request, "result.html",
        {
            "predicted_price": pred,
            "pi_lo": pi_lo,
            "pi_hi": pi_hi,
            "top_contributions": top_contributions,
            "baseline_dollars": baseline,
            "is_pct": False,
            "lin_diag": lin_diag,
            "model_type": model_label,
            "segment": v.segment or "—",
            "n_features": v.n_features,
            "parcel_id": v.parcel_id,
        },
    )