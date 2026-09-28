"""
MAORS Core Clinical Scoring Engine.
Implements the exact clinical scoring rules defined in Table 0 and Table 1 of the MAORS guideline.
"""

from typing import List, Tuple, Dict, Any, Optional
from .models import PatientData, RiskBreakdownItem, AssessmentResult, ClinicalAlert
from .constants import RiskTier, RISK_CATEGORIES, RISK_FACTORS_CONFIG
from .recommendations import generate_clinical_alerts, generate_action_plan

class MAORSEngine:
    """
    Deterministic clinical risk calculation engine for MAORS (Maternal Adverse Outcome Risk Score).
    """

    def __init__(
        self,
        distance_threshold_km: float = 10.0,
        inadequate_weight_gain_threshold_kg: float = 2.0
    ):
        self.distance_threshold_km = distance_threshold_km
        self.inadequate_weight_gain_threshold_kg = inadequate_weight_gain_threshold_kg

    def evaluate(self, patient: PatientData, include_ml: bool = False, ml_predictor: Optional[Any] = None) -> AssessmentResult:
        """
        Evaluate a patient record and return a comprehensive AssessmentResult.
        """
        breakdown: List[RiskBreakdownItem] = []
        total_score = 0

        # Helper to record a factor
        def record_factor(
            factor_id: str,
            factor_name: str,
            domain: str,
            definition: str,
            points: int,
            max_pts: int,
            triggered: bool,
            val: Any,
            note: str
        ):
            nonlocal total_score
            awarded = points if triggered else 0
            total_score += awarded
            breakdown.append(RiskBreakdownItem(
                factor_id=factor_id,
                factor_name=factor_name,
                domain=domain,
                definition=definition,
                points_awarded=awarded,
                max_possible_points=max_pts,
                is_triggered=triggered,
                observed_value=val,
                clinical_note=note
            ))

        # 1. Age (<20 years -> 1 pt; >=35 years -> 1 pt)
        if patient.age < 20:
            record_factor("age_teen", "Age (<20 years)", "Demographic", "<20 years", 1, 1, True, f"{patient.age} yrs", "Adolescent pregnancy risk.")
        elif patient.age >= 35:
            record_factor("age_advanced", "Age (>=35 years)", "Demographic", ">=35 years", 1, 1, True, f"{patient.age} yrs", "Advanced maternal age risk.")
        else:
            record_factor("age", "Age", "Demographic", "20–34 years", 0, 1, False, f"{patient.age} yrs", "Optimal maternal reproductive age.")

        # 2. Previous preterm birth (Yes -> 3 pts)
        record_factor(
            "prev_preterm", "Previous preterm birth", "Obstetric History", "Yes", 3, 3,
            patient.prev_preterm, "Yes" if patient.prev_preterm else "No",
            "Strong historical risk of recurrent preterm delivery." if patient.prev_preterm else "No prior preterm delivery."
        )

        # 3. Previous low-birth-weight baby (Yes -> 2 pts)
        record_factor(
            "prev_lbw", "Previous low-birth-weight baby", "Obstetric History", "Yes", 2, 2,
            patient.prev_lbw, "Yes" if patient.prev_lbw else "No",
            "History of fetal growth restriction or LBW." if patient.prev_lbw else "No prior LBW baby."
        )

        # 4. Previous stillbirth (Yes -> 1 pt)
        record_factor(
            "prev_stillbirth", "Previous stillbirth", "Obstetric History", "Yes", 1, 1,
            patient.prev_stillbirth, "Yes" if patient.prev_stillbirth else "No",
            "History of stillbirth warrants intensified fetal monitoring." if patient.prev_stillbirth else "No prior stillbirth."
        )

        # 5. Previous LSCS (Yes -> 1 pt)
        record_factor(
            "prev_lscs", "Previous LSCS", "Obstetric History", "Yes", 1, 1,
            patient.prev_lscs, "Yes" if patient.prev_lscs else "No",
            "Previous uterine scar; requires delivery mode planning." if patient.prev_lscs else "No previous caesarean scar."
        )

        # 6. Multiple pregnancy (Yes -> 3 pts)
        record_factor(
            "multiple_pregnancy", "Multiple pregnancy", "Current Pregnancy", "Yes", 3, 3,
            patient.multiple, "Yes" if patient.multiple else "No",
            "Multiple gestation has high preterm and preeclampsia risk." if patient.multiple else "Singleton pregnancy."
        )

        # 7. Previous abortions (>=2 -> 1 pt)
        record_factor(
            "prev_abortions", "Previous abortions (>=2)", "Obstetric History", ">=2", 1, 1,
            patient.abortions >= 2, f"{patient.abortions}",
            "Recurrent pregnancy loss history." if patient.abortions >= 2 else "Fewer than 2 previous abortions."
        )

        # 8. Birth interval (<2 years -> 1 pt)
        is_short_interval = (patient.birth_interval is not None) and (patient.birth_interval < 2.0)
        record_factor(
            "birth_interval", "Birth interval (<2 years)", "Obstetric History", "<2 years", 1, 1,
            is_short_interval, f"{patient.birth_interval} yrs" if patient.birth_interval is not None else "N/A",
            "Short interpregnancy interval (<24 months)." if is_short_interval else "Adequate birth spacing (>=2 years)."
        )

        # 9. Gestational age at assessment (<28 weeks -> 1 pt)
        is_early_ga = patient.ga_weeks < 28
        record_factor(
            "ga_assessment", "Gestational age at assessment (<28 weeks)", "Clinical Assessment", "<28 weeks", 1, 1,
            is_early_ga, f"{patient.ga_weeks} wks",
            "Early assessment stage for prophylactic interventions." if is_early_ga else f"Late presentation / assessment at {patient.ga_weeks} weeks."
        )

        # 10. Haemoglobin (<8 g/dL -> 3 pts; 8–9.9 g/dL -> 2 pts)
        if patient.hb < 8.0:
            record_factor("hb_severe", "Haemoglobin (<8 g/dL)", "Laboratory", "<8 g/dL", 3, 3, True, f"{patient.hb} g/dL", "Severe maternal anaemia.")
        elif patient.hb <= 9.9:
            record_factor("hb_moderate", "Haemoglobin (8–9.9 g/dL)", "Laboratory", "8–9.9 g/dL", 2, 3, True, f"{patient.hb} g/dL", "Moderate maternal anaemia.")
        else:
            record_factor("hb_normal", "Haemoglobin", "Laboratory", ">=10 g/dL", 0, 3, False, f"{patient.hb} g/dL", "Normal or mild maternal haemoglobin level.")

        # 11. Systolic BP (130–139 mmHg -> 1 pt; >=140 mmHg -> 3 pts)
        if patient.bp_sys >= 140:
            record_factor("bp_sys_stage2", "Systolic BP (>=140 mmHg)", "Vitals", ">=140 mmHg", 3, 3, True, f"{patient.bp_sys} mmHg", "Hypertensive range systolic BP.")
        elif patient.bp_sys >= 130:
            record_factor("bp_sys_stage1", "Systolic BP (130–139 mmHg)", "Vitals", "130–139 mmHg", 1, 3, True, f"{patient.bp_sys} mmHg", "Pre-hypertensive systolic BP.")
        else:
            record_factor("bp_sys_normal", "Systolic BP", "Vitals", "<130 mmHg", 0, 3, False, f"{patient.bp_sys} mmHg", "Normal systolic blood pressure.")

        # 12. Diastolic BP (80–89 mmHg -> 1 pt; >=90 mmHg -> 3 pts)
        if patient.bp_dia >= 90:
            record_factor("bp_dia_stage2", "Diastolic BP (>=90 mmHg)", "Vitals", ">=90 mmHg", 3, 3, True, f"{patient.bp_dia} mmHg", "Hypertensive range diastolic BP.")
        elif patient.bp_dia >= 80:
            record_factor("bp_dia_stage1", "Diastolic BP (80–89 mmHg)", "Vitals", "80–89 mmHg", 1, 3, True, f"{patient.bp_dia} mmHg", "Pre-hypertensive diastolic BP.")
        else:
            record_factor("bp_dia_normal", "Diastolic BP", "Vitals", "<80 mmHg", 0, 3, False, f"{patient.bp_dia} mmHg", "Normal diastolic blood pressure.")

        # 13. Urine protein (Positive -> 3 pts)
        record_factor(
            "urine_protein", "Urine protein", "Laboratory", "Positive", 3, 3,
            patient.urine_protein, "Positive" if patient.urine_protein else "Negative",
            "Proteinuria detected; major risk factor for preeclampsia." if patient.urine_protein else "No proteinuria."
        )

        # 14. Edema (Present -> 1 pt)
        record_factor(
            "edema", "Edema", "Clinical Signs", "Present", 1, 1,
            patient.edema, "Present" if patient.edema else "Absent",
            "Pathological edema noted." if patient.edema else "No edema."
        )

        # 15. Abnormal USG (Yes -> 2 pts)
        record_factor(
            "usg_abnormal", "Abnormal USG", "Imaging", "Yes", 2, 2,
            patient.usg_abnormal, "Yes" if patient.usg_abnormal else "No",
            "Ultrasound abnormality reported (liquor/placental/fetal)." if patient.usg_abnormal else "Normal obstetric ultrasound."
        )

        # 16. Medical condition (Yes -> 2 pts)
        record_factor(
            "medical_condition", "Medical condition", "Medical History", "Yes", 2, 2,
            patient.medical_condition, "Yes" if patient.medical_condition else "No",
            "Pre-existing medical comorbidity present." if patient.medical_condition else "No known medical comorbidities."
        )

        # 17. Inadequate weight gain (Yes -> 2 pts)
        is_inadequate_wg = (patient.weight_gain is not None) and (patient.weight_gain <= self.inadequate_weight_gain_threshold_kg)
        record_factor(
            "inadequate_weight_gain", "Inadequate weight gain", "Nutritional", "Yes", 2, 2,
            is_inadequate_wg, f"{patient.weight_gain} kg" if patient.weight_gain is not None else "N/A",
            f"Poor gestational weight gain (<={self.inadequate_weight_gain_threshold_kg} kg)." if is_inadequate_wg else "Satisfactory gestational weight gain."
        )

        # 18. ANC visits (<4 visits -> 2 pts)
        is_suboptimal_anc = patient.anc_visits < 4
        record_factor(
            "anc_visits", "ANC visits (<4 visits)", "Health System", "<4 visits", 2, 2,
            is_suboptimal_anc, f"{patient.anc_visits} visits",
            "Fewer than 4 standard ANC visits completed." if is_suboptimal_anc else "Adequate ANC attendance (>=4 visits)."
        )

        # 19. IFA (Not taking -> 1 pt)
        is_not_taking_ifa = not patient.ifa
        record_factor(
            "ifa_noncompliance", "IFA (Iron Folic Acid)", "Nutritional / Prophylaxis", "Not taking", 1, 1,
            is_not_taking_ifa, "Not taking" if is_not_taking_ifa else "Taking regularly",
            "Non-compliance with prophylactic iron-folic acid." if is_not_taking_ifa else "Compliant with IFA."
        )

        # 20. TT (Not received/incomplete -> 1 pt)
        is_tt_incomplete = not patient.tt
        record_factor(
            "tt_incomplete", "TT (Tetanus Toxoid)", "Immunization", "Not received/incomplete", 1, 1,
            is_tt_incomplete, "Incomplete/Not received" if is_tt_incomplete else "Received",
            "Inadequate tetanus toxoid immunization." if is_tt_incomplete else "Up-to-date TT immunization."
        )

        # 21. Poor diet (Yes -> 1 pt; diet == False)
        is_poor_diet = not patient.diet
        record_factor(
            "poor_diet", "Poor diet", "Nutritional", "Yes", 1, 1,
            is_poor_diet, "Poor/Inadequate" if is_poor_diet else "Adequate",
            "Inadequate dietary quality and diversity." if is_poor_diet else "Nutritious balanced diet reported."
        )

        # 22. Tobacco use (Yes -> 2 pts)
        record_factor(
            "tobacco_use", "Tobacco use", "Behavioral", "Yes", 2, 2,
            patient.tobacco, "Yes" if patient.tobacco else "No",
            "Active tobacco exposure (smoking or chewing)." if patient.tobacco else "No tobacco use."
        )

        # 23. Long distance to facility (Yes -> 1 pt)
        is_long_dist = (patient.distance is not None) and (patient.distance >= self.distance_threshold_km)
        record_factor(
            "long_distance", "Long distance to facility", "Socio-Environmental", "Yes", 1, 1,
            is_long_dist, f"{patient.distance} km" if patient.distance is not None else "N/A",
            f"Distance to emergency facility >={self.distance_threshold_km} km." if is_long_dist else "Resides near obstetric facility."
        )

        # 24. Transport difficulty (Yes -> 1 pt)
        record_factor(
            "transport_difficulty", "Transport difficulty", "Socio-Environmental", "Yes", 1, 1,
            patient.transport, "Yes" if patient.transport else "No",
            "Difficult emergency transit reported." if patient.transport else "Accessible transportation."
        )

        # 25. Inadequate support (Yes -> 1 pt; support == False)
        is_inadequate_supp = not patient.support
        record_factor(
            "inadequate_support", "Inadequate support", "Social", "Yes", 1, 1,
            is_inadequate_supp, "Inadequate" if is_inadequate_supp else "Adequate",
            "Lacks primary family/caregiver support." if is_inadequate_supp else "Adequate social/family support."
        )

        # 26. Delay in seeking/receiving care (Yes -> 1 pt)
        record_factor(
            "delay_in_care", "Delay in seeking/receiving care", "Health System / Behavioral", "Yes", 1, 1,
            patient.delay, "Yes" if patient.delay else "No",
            "History of care-seeking or facility delays." if patient.delay else "No critical care delays reported."
        )

        # Determine Risk Tier based on Table 1
        if total_score <= 5:
            tier = RiskTier.LOW
        elif total_score <= 10:
            tier = RiskTier.MODERATE
        elif total_score <= 15:
            tier = RiskTier.HIGH
        else:
            tier = RiskTier.VERY_HIGH

        cat_meta = RISK_CATEGORIES[tier]

        # Clinical Alerts and Action Plan
        alerts = generate_clinical_alerts(patient, total_score)
        action_plan = generate_action_plan(patient, total_score, tier, breakdown)

        # ML Predictions (optional)
        ml_pred = None
        if include_ml and ml_predictor is not None:
            try:
                ml_pred = ml_predictor.predict(patient)
            except Exception as e:
                # Silently catch or log ML error so clinical deterministic calculation never fails
                pass

        vitals_summary = {
            "age": patient.age,
            "ga_weeks": patient.ga_weeks,
            "blood_pressure": f"{patient.bp_sys}/{patient.bp_dia} mmHg",
            "map": patient.get_map(),
            "pulse_pressure": patient.get_pulse_pressure(),
            "hb": f"{patient.hb} g/dL",
            "bmi": patient.get_bmi(),
            "weight_gain": f"{patient.weight_gain} kg" if patient.weight_gain is not None else None,
            "fundal_lag": patient.get_fundal_lag()
        }

        return AssessmentResult(
            patient_id=str(patient.id or "P001"),
            total_score=total_score,
            risk_tier=tier.value,
            risk_category_label=cat_meta.label,
            badge=cat_meta.badge,
            color_hex=cat_meta.color,
            suggested_interpretation=cat_meta.action,
            surveillance_interval=cat_meta.surveillance_interval,
            referral_level=cat_meta.referral_level,
            clinical_alerts=alerts,
            action_plan=action_plan,
            risk_breakdown=breakdown,
            ml_prediction=ml_pred,
            vitals_summary=vitals_summary
        )
