"""
Clinical care pathways, evidence-based recommendations, and protocol generators for MAORS.
"""

from typing import List
from .models import PatientData, ClinicalAlert, RiskBreakdownItem
from .constants import RiskTier

def generate_clinical_alerts(patient: PatientData, score: int) -> List[ClinicalAlert]:
    """Identify high-priority, acute clinical red flags."""
    alerts = []

    # 1. Preeclampsia / Hypertensive Emergency
    if patient.bp_sys >= 140 or patient.bp_dia >= 90:
        if patient.urine_protein:
            alerts.append(ClinicalAlert(
                title="Preeclampsia Alert",
                severity="CRITICAL",
                category="Obstetric / Hypertensive Crisis",
                message=f"Severe Hypertension ({patient.bp_sys}/{patient.bp_dia} mmHg) combined with Proteinuria detected.",
                action_required="Urgent specialist evaluation; consider admission, antihypertensive therapy, seizure prophylaxis (MgSO4 protocol), and fetal biophysical profile."
            ))
        else:
            alerts.append(ClinicalAlert(
                title="Gestational Hypertension Alert",
                severity="WARNING",
                category="Vascular / Blood Pressure",
                message=f"Blood pressure elevated ({patient.bp_sys}/{patient.bp_dia} mmHg) without immediate proteinuria.",
                action_required="Perform repeat BP measurements, 24h urine protein test, liver/renal function panels, and weekly Doppler ultrasound."
            ))

    # 2. Severe Anaemia Alert
    if patient.hb < 8.0:
        alerts.append(ClinicalAlert(
            title="Severe Maternal Anaemia",
            severity="CRITICAL",
            category="Hematological",
            message=f"Haemoglobin level is severely low ({patient.hb} g/dL).",
            action_required="Investigate underlying etiology (iron deficiency, hemoglobinopathy); initiate injectable iron (IV ferric carboxymaltose) or arrange blood transfusion workup if symptomatic."
        ))
    elif patient.hb < 10.0:
        alerts.append(ClinicalAlert(
            title="Moderate Maternal Anaemia",
            severity="WARNING",
            category="Hematological",
            message=f"Haemoglobin level is low ({patient.hb} g/dL).",
            action_required="Prescribe therapeutic oral IFA (100mg elemental iron twice daily), assess dietary intake and stool deworming."
        ))

    # 3. Preterm Birth High Risk Alert
    if patient.prev_preterm or patient.multiple:
        reasons = []
        if patient.prev_preterm:
            reasons.append("History of prior preterm birth")
        if patient.multiple:
            reasons.append("Multiple gestation")
        alerts.append(ClinicalAlert(
            title="High Preterm Delivery Threat",
            severity="WARNING",
            category="Obstetric",
            message=f"Elevated risk of preterm labor due to: {', '.join(reasons)}.",
            action_required="Serial transvaginal cervical length screening, progesterone supplementation if indicated, and plan corticosteroid coverage (Dexamethasone/Betamethasone) if preterm labor ensues."
        ))

    # 4. Fetal Growth Restriction / FGR / Malnutrition
    if patient.weight_gain is not None and patient.weight_gain <= 2.0 and patient.ga_weeks >= 24:
        alerts.append(ClinicalAlert(
            title="Fetal Growth Restriction / Inadequate Weight Gain",
            severity="WARNING",
            category="Nutritional & Fetal Growth",
            message=f"Total weight gain is only {patient.weight_gain} kg at {patient.ga_weeks} weeks.",
            action_required="Serial obstetric ultrasound with umbilical artery Doppler, intensive nutritional counseling, and calorie-dense balanced protein supplementation."
        ))

    # 5. Decreased Fetal Movement
    if not patient.fetal_movement and patient.ga_weeks >= 24:
        alerts.append(ClinicalAlert(
            title="Decreased Fetal Movement (DFM) Alert",
            severity="CRITICAL",
            category="Fetal Surveillance",
            message="Absence or marked reduction in fetal movements reported.",
            action_required="Immediate cardiotocography (NST) and urgent obstetric ultrasound to assess fetal biophysical profile and amniotic fluid index."
        ))

    # 6. Malpresentation in Late Third Trimester
    if patient.baby_position == "NotHead" and patient.ga_weeks >= 36:
        alerts.append(ClinicalAlert(
            title="Fetal Malpresentation at Term/Near Term",
            severity="WARNING",
            category="Delivery Planning",
            message=f"Fetus is non-cephalic at {patient.ga_weeks} weeks.",
            action_required="Confirm fetal position by ultrasound; evaluate for External Cephalic Version (ECV) or schedule elective caesarean delivery at tertiary facility."
        ))

    # 7. Socio-Environmental Delay & Transit Vulnerability
    if (patient.distance and patient.distance >= 15.0) or patient.transport or patient.delay:
        alerts.append(ClinicalAlert(
            title="Access & Transit Vulnerability",
            severity="INFO",
            category="Socio-Environmental",
            message="Patient faces significant distance, lack of transport, or care delays.",
            action_required="Establish institutional birth preparedness plan, pre-identify emergency transport/ambulance helpline, and consider maternity waiting home near delivery date."
        ))

    return alerts


def generate_action_plan(patient: PatientData, score: int, tier: RiskTier, breakdown: List[RiskBreakdownItem]) -> List[str]:
    """Build a step-by-step actionable clinical checklist tailored to the patient's risk profile."""
    actions = []

    # Category Level Foundation
    if tier == RiskTier.LOW:
        actions.append("Schedule standard ANC follow-up according to national guidelines (minimum 4-8 visits).")
        actions.append("Continue routine daily prophylaxis: Iron & Folic Acid (IFA) and Calcium supplementation.")
        actions.append("Screen for danger signs (bleeding, swelling, headache, vision changes) at each routine visit.")
    elif tier == RiskTier.MODERATE:
        actions.append("Shorten ANC visit intervals (every 2 weeks up to 32 weeks, weekly thereafter).")
        actions.append("Conduct comprehensive nutritional and lifestyle counselling.")
        actions.append("Confirm two doses of Tetanus Toxoid / Td vaccination completed.")
        actions.append("Repeat routine blood pressure and urine protein dipstick at every visit.")
    elif tier == RiskTier.HIGH:
        actions.append("Refer to Medical Officer / Obstetrician for clinical case review and individualized birth plan.")
        actions.append("Schedule targeted obstetrical ultrasound for fetal growth, amniotic fluid volume, and Doppler velocimetry.")
        actions.append("Enhanced maternal lab workup: Complete Blood Count, Renal Function, Liver Function, and Blood Sugar.")
        actions.append("Designate institutional delivery facility with 24x7 emergency surgical and neonatal capabilities.")
    else:  # VERY_HIGH
        actions.append("URGENT: Arrange immediate specialist evaluation / referral to Tertiary Care Medical Centre.")
        actions.append("Ensure 24x7 blood bank availability and Neonatal Intensive Care Unit (NICU) readiness.")
        actions.append("Initiate disease-specific emergency protocols (e.g. anti-hypertensive control, IV iron, corticosteroid coverage).")
        actions.append("Formulate an urgent transport and emergency accompaniment plan with community health worker (ASHA/ANM).")

    # Specific Triggered Actions
    for item in breakdown:
        if not item.is_triggered:
            continue
        if item.factor_id in ['hb_severe', 'hb_moderate']:
            actions.append(f"Anaemia Care: {item.clinical_note} Ensure therapeutic iron administration and repeat Hb in 4 weeks.")
        elif item.factor_id in ['bp_sys_stage2', 'bp_dia_stage2', 'urine_protein']:
            actions.append("Hypertension/Proteinuria Protocol: Monitor BP twice daily, track 24-hr urine protein, watch for symptoms of severe preeclampsia (headache, epigastric pain, visual scotoma).")
        elif item.factor_id == 'tobacco_use':
            actions.append("Substance Cessation: Provide structured tobacco cessation counseling and support resources.")
        elif item.factor_id == 'inadequate_weight_gain':
            actions.append("Nutrition Support: Refer for supplementary nutrition (ICDS/PMMVY schemes) and dietary caloric boost.")
        elif item.factor_id in ['long_distance', 'transport_difficulty', 'inadequate_support']:
            actions.append("Social & Logistics: Coordinate with local health worker to link patient with maternal transit vouchers and institutional birth lodging.")

    # Deduplicate while preserving order
    seen = set()
    deduped_actions = []
    for act in actions:
        if act not in seen:
            seen.add(act)
            deduped_actions.append(act)

    return deduped_actions
