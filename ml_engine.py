"""
Machine Learning and Hybrid AI Predictor for Maternal Adverse Pregnancy Outcomes.
Works in synergy with the deterministic MAORS clinical scoring engine.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from .models import PatientData, MLPredictionResult

class MAORSMLPredictor:
    """
    Predicts probabilistic risks of specific maternal adverse outcomes:
    - Preterm Birth (<37 weeks)
    - Low Birth Weight (<2.5 kg)
    - Preeclampsia / Gestational Hypertensive Disorders
    - Composite Adverse Pregnancy Outcome
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            # Default to ../models or ./models relative to file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.feature_cols: List[str] = []
        self.is_loaded = False
        self._load_models()

    def _load_models(self):
        try:
            feat_path = os.path.join(self.models_dir, "feature_cols.joblib")
            if os.path.exists(feat_path):
                self.feature_cols = joblib.load(feat_path)
                
            stats_path = os.path.join(self.models_dir, "feature_stats.joblib")
            if os.path.exists(stats_path):
                self.feature_stats = joblib.load(stats_path)
            else:
                self.feature_stats = None
            
            for target in ['Outcome_Preterm', 'Outcome_LBW', 'Outcome_Preeclampsia']:
                model_path = os.path.join(self.models_dir, f"{target}_model.joblib")
                if os.path.exists(model_path):
                    self.models[target] = joblib.load(model_path)
            
            if len(self.models) == 3 and len(self.feature_cols) > 0:
                self.is_loaded = True
        except Exception as e:
            self.is_loaded = False

    def extract_features(self, patient: PatientData) -> pd.DataFrame:
        """Transforms a PatientData object into a DataFrame row matching ML feature schema."""
        height = patient.height or 152.0
        weight = patient.weight or 58.0
        height_m = height / 100.0
        bmi = weight / (height_m ** 2) if height_m > 0 else 24.0
        
        bp_sys = patient.bp_sys
        bp_dia = patient.bp_dia
        map_val = (2.0 * bp_dia + bp_sys) / 3.0
        pulse_pressure = bp_sys - bp_dia
        
        ga_weeks = patient.ga_weeks
        fundal_height = patient.fundal_height or ga_weeks
        fundal_lag = ga_weeks - fundal_height
        
        baby_pos_nothead = 1 if (patient.baby_position and patient.baby_position.strip().lower() != 'head') else 0

        row = {
            'Age': patient.age,
            'Height': height,
            'Weight': weight,
            'Gravida': patient.gravida or 1,
            'GA_weeks': ga_weeks,
            'Prev_Stillbirth': int(patient.prev_stillbirth),
            'Prev_Preterm': int(patient.prev_preterm),
            'Prev_LBW': int(patient.prev_lbw),
            'Prev_LSCS': int(patient.prev_lscs),
            'Abortions': patient.abortions,
            'Birth_Interval': patient.birth_interval if patient.birth_interval is not None else 2.5,
            'Hb': patient.hb,
            'BP_sys': bp_sys,
            'BP_dia': bp_dia,
            'Sugar': patient.sugar if patient.sugar is not None else 100.0,
            'Weight_Gain': patient.weight_gain if patient.weight_gain is not None else 5.0,
            'Edema': int(patient.edema),
            'Fetal_Movement': int(patient.fetal_movement),
            'Fundal_Height': fundal_height,
            'Multiple': int(patient.multiple),
            'Urine_Protein': int(patient.urine_protein),
            'USG_Abnormal': int(patient.usg_abnormal),
            'Medical_Condition': int(patient.medical_condition),
            'ANC_Visits': patient.anc_visits,
            'TT': int(patient.tt),
            'IFA': int(patient.ifa),
            'Diet': int(patient.diet),
            'Tobacco': int(patient.tobacco),
            'Distance': patient.distance if patient.distance is not None else 10.0,
            'Transport': int(patient.transport),
            'Support': int(patient.support),
            'Delay': int(patient.delay),
            'BMI': bmi,
            'MAP': map_val,
            'Pulse_Pressure': pulse_pressure,
            'Fundal_Lag': fundal_lag,
            'Baby_Position_NotHead': baby_pos_nothead
        }

        df_row = pd.DataFrame([row])
        if self.feature_cols:
            # Ensure all required columns exist in expected order
            for col in self.feature_cols:
                if col not in df_row.columns:
                    df_row[col] = 0
            df_row = df_row[self.feature_cols]
        return df_row

    def predict(self, patient: PatientData) -> MLPredictionResult:
        """Generate outcome probabilities and drivers."""
        top_drivers = []
        
        if self.is_loaded:
            X = self.extract_features(patient)
            
            # Predict probabilities
            p_preterm = float(self.models['Outcome_Preterm'].predict_proba(X)[0][1])
            p_lbw = float(self.models['Outcome_LBW'].predict_proba(X)[0][1])
            p_preecl = float(self.models['Outcome_Preeclampsia'].predict_proba(X)[0][1])
            
            # Composite adverse probability: P(Preterm OR LBW OR Preeclampsia)
            # 1 - (1 - P1)*(1 - P2)*(1 - P3) assuming conditional independence upper bound or model-based
            p_composite = 1.0 - ((1.0 - p_preterm) * (1.0 - p_lbw) * (1.0 - p_preecl))
            p_composite = min(1.0, max(0.0, p_composite))
            
            # Dynamic Driver Calculation (Approximation)
            max_risk_target = 'Outcome_Preterm'
            max_risk_prob = p_preterm
            if p_lbw > max_risk_prob:
                max_risk_target = 'Outcome_LBW'
                max_risk_prob = p_lbw
            if p_preecl > max_risk_prob:
                max_risk_target = 'Outcome_Preeclampsia'
                max_risk_prob = p_preecl

            if hasattr(self, 'feature_stats') and self.feature_stats:
                pipeline = self.models[max_risk_target]
                if hasattr(pipeline, 'named_steps') and 'gb' in pipeline.named_steps:
                    gb_model = pipeline.named_steps['gb']
                    importances = gb_model.feature_importances_
                    
                    contributions = []
                    x_dict = X.iloc[0].to_dict()
                    means = self.feature_stats['mean']
                    stds = self.feature_stats['std']
                    
                    for i, col in enumerate(self.feature_cols):
                        val = x_dict.get(col, 0)
                        mean_val = means.get(col, 0)
                        std_val = stds.get(col, 1)
                        if std_val == 0:
                            std_val = 1
                            
                        # Standardized deviation
                        z_score = (val - mean_val) / std_val
                        # Weighted by feature importance
                        driver_score = importances[i] * abs(z_score)
                        
                        if driver_score > 0.01:
                            direction = "Elevated" if z_score > 0 else "Lowered"
                            contributions.append((driver_score, f"{direction} {col} (Target: {max_risk_target.replace('Outcome_', '')})"))
                    
                    contributions.sort(key=lambda x: x[0], reverse=True)
                    top_drivers = [desc for score, desc in contributions[:4]]
                    
            if not top_drivers:
                top_drivers.append("No isolated critical drivers; combined multi-factor risk.")

            return MLPredictionResult(
                preterm_risk_prob=round(p_preterm, 4),
                lbw_risk_prob=round(p_lbw, 4),
                preeclampsia_risk_prob=round(p_preecl, 4),
                composite_adverse_prob=round(p_composite, 4),
                model_version="1.0.0-gradient-boost",
                confidence_tier="High",
                top_driving_factors=top_drivers[:4]
            )
        else:
            # Fallback heuristic calculation if models are unpickled or offline
            p_preterm = 0.85 if (patient.prev_preterm or patient.multiple or patient.hb < 8.0) else 0.15
            p_lbw = 0.90 if (not patient.diet or (patient.weight_gain is not None and patient.weight_gain <= 2) or patient.hb < 8.0) else 0.20
            p_preecl = 0.95 if (patient.urine_protein and patient.bp_sys >= 140) else (0.40 if patient.bp_sys >= 130 else 0.10)
            p_composite = 1.0 - ((1.0 - p_preterm) * (1.0 - p_lbw) * (1.0 - p_preecl))
            
            return MLPredictionResult(
                preterm_risk_prob=round(p_preterm, 4),
                lbw_risk_prob=round(p_lbw, 4),
                preeclampsia_risk_prob=round(p_preecl, 4),
                composite_adverse_prob=round(p_composite, 4),
                model_version="1.0.0-heuristic-fallback",
                confidence_tier="Estimated",
                top_driving_factors=["Clinical Rule Heuristics (ML models not initialized)"]
            )
