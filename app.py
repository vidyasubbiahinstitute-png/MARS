"""
MAORS (Maternal Adverse Outcome Risk Score) - FastAPI Backend Server.
Provides high-performance REST APIs for real-time assessment, batch processing, and rule exploration.
"""

import sys
import os
import io
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List

# Ensure current directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from maors import (
    MAORSEngine,
    MAORSMLPredictor,
    MAORSPipeline,
    PatientData,
    AssessmentResult,
    RISK_CATEGORIES,
    RISK_FACTORS_CONFIG,
    RiskTier
)
from maors.validator import PatientValidator

app = FastAPI(
    title="MAORS - Maternal Adverse Outcome Risk Score API",
    description="High-performance clinical risk scoring and predictive intelligence platform for maternal health.",
    version="1.0.0"
)

# CORS Middleware for seamless browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
engine = MAORSEngine()
ml_predictor = MAORSMLPredictor()
pipeline = MAORSPipeline(engine=engine, ml_predictor=ml_predictor)

# Load and pre-score sample dataset into memory for instant demo & testing
DATASET_PATH = os.path.join(BASE_DIR, "maternal_ai_dataset.xlsx")
sample_dataset_df = None
cached_scored_df = None
cached_summary = None

if os.path.exists(DATASET_PATH):
    try:
        sample_dataset_df = pd.read_excel(DATASET_PATH)
        cached_scored_df, cached_summary = pipeline.process_dataset(sample_dataset_df, include_ml=True)
    except Exception as e:
        print(f"Error loading sample dataset: {e}")

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "version": "1.0.0",
        "ml_models_loaded": ml_predictor.is_loaded,
        "sample_dataset_loaded": sample_dataset_df is not None,
        "cached_cohort_size": len(cached_scored_df) if cached_scored_df is not None else 0
    }

@app.get("/api/config/rules")
def get_rules_config():
    """Returns all 26 risk factors and 4 risk tiers configuration."""
    tiers_dict = {}
    for tier_enum, cat in RISK_CATEGORIES.items():
        tiers_dict[tier_enum.value] = cat.to_dict()

    return {
        "statement": "A simple clinical risk score for early identification of pregnant women at risk of adverse pregnancy outcomes.",
        "risk_tiers": tiers_dict,
        "risk_factors": RISK_FACTORS_CONFIG,
        "total_factors": len(RISK_FACTORS_CONFIG)
    }

@app.post("/api/assess")
def assess_patient(patient_dict: Dict[str, Any]):
    """
    Assess a single patient record.
    Returns complete clinical risk score, risk tier, alerts, action plan, and ML outcome probabilities.
    """
    patient, errors, warnings = PatientValidator.validate_and_sanitize(patient_dict)
    if patient is None:
        raise HTTPException(status_code=400, detail={"errors": errors})

    result = engine.evaluate(patient, include_ml=True, ml_predictor=ml_predictor)
    res_dict = result.to_dict()
    res_dict["validation_warnings"] = warnings
    return res_dict

@app.post("/api/whatif")
def simulate_whatif(payload: Dict[str, Any]):
    """
    Simulate targeted clinical interventions on a baseline patient record.
    Returns baseline vs simulated scores, risk tier shifts, and ML outcome risk deltas.
    """
    baseline_data = payload.get("baseline", {})
    interventions = payload.get("interventions", {})

    p_base, errs_base, _ = PatientValidator.validate_and_sanitize(baseline_data)
    if p_base is None:
        raise HTTPException(status_code=400, detail={"errors": errs_base})

    base_result = engine.evaluate(p_base, include_ml=True, ml_predictor=ml_predictor)

    # Build simulated patient data
    sim_dict = baseline_data.copy()
    if "hb" in interventions:
        sim_dict["hb"] = interventions["hb"]
    if "bp_sys" in interventions:
        sim_dict["bp_sys"] = interventions["bp_sys"]
        sim_dict["bp_dia"] = interventions.get("bp_dia", min(sim_dict.get("bp_dia", 80), int(interventions["bp_sys"] * 0.65)))
        if sim_dict["bp_sys"] < 140:
            sim_dict["urine_protein"] = False
    if "diet" in interventions:
        sim_dict["diet"] = bool(interventions["diet"])
    if "transport" in interventions:
        sim_dict["transport"] = bool(interventions["transport"])
    if "ifa" in interventions:
        sim_dict["ifa"] = bool(interventions["ifa"])
    if "tobacco" in interventions:
        sim_dict["tobacco"] = bool(interventions["tobacco"])
    if "anc_visits" in interventions:
        sim_dict["anc_visits"] = int(interventions["anc_visits"])

    p_sim, errs_sim, _ = PatientValidator.validate_and_sanitize(sim_dict)
    if p_sim is None:
        raise HTTPException(status_code=400, detail={"errors": errs_sim})

    sim_result = engine.evaluate(p_sim, include_ml=True, ml_predictor=ml_predictor)

    base_dict = base_result.to_dict()
    sim_dict_res = sim_result.to_dict()

    score_delta = sim_result.total_score - base_result.total_score

    # ML deltas
    base_ml = base_dict.get("ml_prediction") or base_dict.get("ml_predictions") or {}
    sim_ml = sim_dict_res.get("ml_prediction") or sim_dict_res.get("ml_predictions") or {}

    preterm_delta = round((sim_ml.get("preterm_risk_prob", 0) - base_ml.get("preterm_risk_prob", 0)) * 100, 1)
    lbw_delta = round((sim_ml.get("lbw_risk_prob", 0) - base_ml.get("lbw_risk_prob", 0)) * 100, 1)
    preecl_delta = round((sim_ml.get("preeclampsia_risk_prob", 0) - base_ml.get("preeclampsia_risk_prob", 0)) * 100, 1)
    composite_delta = round((sim_ml.get("composite_adverse_prob", 0) - base_ml.get("composite_adverse_prob", 0)) * 100, 1)

    return {
        "baseline": base_dict,
        "simulated": sim_dict_res,
        "score_delta": score_delta,
        "tier_improved": sim_result.risk_tier != base_result.risk_tier and score_delta < 0,
        "ml_deltas_percent": {
            "preterm": preterm_delta,
            "lbw": lbw_delta,
            "preeclampsia": preecl_delta,
            "composite": composite_delta
        }
    }

@app.post("/api/batch")
async def batch_process_dataset(file: UploadFile = File(...)):
    """
    Upload an Excel (.xlsx/.xls) or CSV file for instant cohort batch scoring.
    """
    try:
        contents = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an Excel (.xlsx, .xls) or CSV file.")

        scored_df, summary = pipeline.process_dataset(df, include_ml=True)
        
        # Convert first 100 rows to JSON records for frontend display
        records = scored_df.head(100).to_dict(orient='records')
        # Clean NaNs for JSON serialization
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    r[k] = None

        return {
            "filename": file.filename,
            "total_rows": len(scored_df),
            "summary": summary,
            "sample_records": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")

@app.get("/api/dataset/sample")
def get_sample_records(limit: int = 100):
    """Returns scored patient cases and cohort summary from the 500-patient maternal dataset."""
    if cached_scored_df is not None and cached_summary is not None:
        records = cached_scored_df.head(limit).to_dict(orient='records')
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    r[k] = None
        return {
            "count": len(records),
            "total_rows": len(cached_scored_df),
            "summary": cached_summary,
            "records": records
        }

    if sample_dataset_df is None:
        raise HTTPException(status_code=404, detail="Sample dataset not available.")
    
    scored_df, summary = pipeline.process_dataset(sample_dataset_df, include_ml=True)
    records = scored_df.head(limit).to_dict(orient='records')
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                r[k] = None
    return {
        "count": len(records),
        "total_rows": len(scored_df),
        "summary": summary,
        "records": records
    }

@app.get("/api/dataset/audit")
def get_dataset_audit():
    """Runs full audit on the 500 patient cohort."""
    if cached_summary is not None:
        return cached_summary
    if sample_dataset_df is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    scored_df, summary = pipeline.process_dataset(sample_dataset_df, include_ml=True)
    return summary

@app.get("/api/export/sample-scored")
def export_sample_scored(format: str = "xlsx"):
    """Download full scored dataset as Excel or CSV."""
    df_to_export = cached_scored_df
    if df_to_export is None:
        if sample_dataset_df is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        df_to_export, _ = pipeline.process_dataset(sample_dataset_df, include_ml=True)
    
    if format.lower() == "csv":
        stream = io.StringIO()
        df_to_export.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=MAORS_Scored_Dataset.csv"
        return response
    else:
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            df_to_export.to_excel(writer, index=False, sheet_name="MAORS_Scored_Cohort")
        stream.seek(0)
        response = StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.headers["Content-Disposition"] = "attachment; filename=MAORS_Scored_Dataset.xlsx"
        return response

# Serve Web UI HTML
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(ui_html_path):
        with open(ui_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>MAORS UI is building...</h1>"

if __name__ == "__main__":
    import uvicorn
    print("Starting MAORS Web Application on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
