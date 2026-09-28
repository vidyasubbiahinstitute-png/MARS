"""
MAORS Batch Pipeline and Dataset Evaluation Processor.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from .models import PatientData, AssessmentResult
from .engine import MAORSEngine
from .ml_engine import MAORSMLPredictor
from .validator import PatientValidator

class MAORSPipeline:
    """
    End-to-end dataset processor for batch screening, auditing, and clinical scoring.
    """

    def __init__(
        self,
        engine: Optional[MAORSEngine] = None,
        ml_predictor: Optional[MAORSMLPredictor] = None
    ):
        self.engine = engine or MAORSEngine()
        self.ml_predictor = ml_predictor or MAORSMLPredictor()

    def process_patient(self, raw_input: Dict[str, Any], include_ml: bool = True) -> AssessmentResult:
        """Processes a single patient record dictionary."""
        patient, errors, warnings = PatientValidator.validate_and_sanitize(raw_input)
        if patient is None:
            raise ValueError(f"Invalid patient input: {'; '.join(errors)}")
        return self.engine.evaluate(patient, include_ml=include_ml, ml_predictor=self.ml_predictor)

    def process_dataset(
        self,
        filepath_or_df: Union[str, pd.DataFrame],
        sheet_name: Optional[str] = None,
        include_ml: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process an entire dataset from an Excel file, CSV file, or DataFrame.
        Returns:
            - scored_df: Enhanced DataFrame containing scores, risk categories, alerts, and ML probabilities
            - summary_metrics: Cohort-level summary metrics and risk distribution
        """
        if isinstance(filepath_or_df, str):
            if filepath_or_df.endswith('.xlsx') or filepath_or_df.endswith('.xls'):
                df = pd.read_excel(filepath_or_df, sheet_name=sheet_name or 0)
            else:
                df = pd.read_csv(filepath_or_df)
        else:
            df = filepath_or_df.copy()

        scores = []
        risk_tiers = []
        risk_labels = []
        badges = []
        interpretations = []
        surveillance_plans = []
        referral_levels = []
        alert_counts = []
        alert_titles = []
        
        ml_preterm = []
        ml_lbw = []
        ml_preecl = []
        ml_composite = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            patient, _, _ = PatientValidator.validate_and_sanitize(row_dict)
            if patient is None:
                patient = PatientData(id=f"Row_{idx+1}")

            result = self.engine.evaluate(patient, include_ml=include_ml, ml_predictor=self.ml_predictor)
            
            scores.append(result.total_score)
            risk_tiers.append(result.risk_tier)
            risk_labels.append(result.risk_category_label)
            badges.append(result.badge)
            interpretations.append(result.suggested_interpretation)
            surveillance_plans.append(result.surveillance_interval)
            referral_levels.append(result.referral_level)
            alert_counts.append(len(result.clinical_alerts))
            alert_titles.append("; ".join([a.title for a in result.clinical_alerts]))

            if result.ml_prediction:
                ml_preterm.append(result.ml_prediction.preterm_risk_prob)
                ml_lbw.append(result.ml_prediction.lbw_risk_prob)
                ml_preecl.append(result.ml_prediction.preeclampsia_risk_prob)
                ml_composite.append(result.ml_prediction.composite_adverse_prob)
            else:
                ml_preterm.append(np.nan)
                ml_lbw.append(np.nan)
                ml_preecl.append(np.nan)
                ml_composite.append(np.nan)

        scored_df = df.copy()
        scored_df['MAORS_Score'] = scores
        scored_df['MAORS_Risk_Tier'] = risk_tiers
        scored_df['MAORS_Risk_Label'] = risk_labels
        scored_df['MAORS_Badge'] = badges
        scored_df['MAORS_Clinical_Action'] = interpretations
        scored_df['MAORS_Surveillance'] = surveillance_plans
        scored_df['MAORS_Referral_Level'] = referral_levels
        scored_df['MAORS_Alert_Count'] = alert_counts
        scored_df['MAORS_Alerts'] = alert_titles

        if include_ml:
            scored_df['ML_Prob_Preterm'] = ml_preterm
            scored_df['ML_Prob_LBW'] = ml_lbw
            scored_df['ML_Prob_Preeclampsia'] = ml_preecl
            scored_df['ML_Prob_Composite_Adverse'] = ml_composite

        # Cohort summary statistics
        total_patients = len(scored_df)
        tier_counts = scored_df['MAORS_Risk_Tier'].value_counts().to_dict()
        tier_percents = {k: round((v / total_patients) * 100, 2) for k, v in tier_counts.items()}

        summary_metrics = {
            "total_screened": total_patients,
            "mean_score": round(float(scored_df['MAORS_Score'].mean()), 2),
            "median_score": round(float(scored_df['MAORS_Score'].median()), 2),
            "min_score": int(scored_df['MAORS_Score'].min()),
            "max_score": int(scored_df['MAORS_Score'].max()),
            "tier_distribution": tier_counts,
            "tier_percentages": tier_percents,
            "total_clinical_alerts_flagged": int(scored_df['MAORS_Alert_Count'].sum())
        }

        # If actual outcome columns exist in the dataset, compute stratification validation
        target_cols = [c for c in ['Outcome_Preterm', 'Outcome_LBW', 'Outcome_Preeclampsia'] if c in scored_df.columns]
        if target_cols:
            outcome_stratification = {}
            for tier in ['LOW', 'MODERATE', 'HIGH', 'VERY_HIGH']:
                sub = scored_df[scored_df['MAORS_Risk_Tier'] == tier]
                n_tier = len(sub)
                if n_tier > 0:
                    strat = {"n": n_tier, "percentage_of_cohort": round(n_tier / total_patients * 100, 1)}
                    for tgt in target_cols:
                        strat[tgt] = round(float(sub[tgt].mean()) * 100, 1)
                    if len(target_cols) >= 2:
                        any_adv = ((sub['Outcome_Preterm'] == 1) | (sub['Outcome_LBW'] == 1) | (sub['Outcome_Preeclampsia'] == 1)).mean() * 100
                        strat["Any_Adverse_Outcome"] = round(float(any_adv), 1)
                    outcome_stratification[tier] = strat
            summary_metrics["outcome_stratification"] = outcome_stratification

        return scored_df, summary_metrics
