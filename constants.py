"""
Constants, configurations, and clinical definitions for MAORS (Maternal Adverse Outcome Risk Score).
Extracted and calibrated against clinical protocols and scoring guidelines.
"""

from enum import Enum
from typing import Dict, Any, List

class RiskTier(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class RiskCategory:
    def __init__(
        self,
        tier: RiskTier,
        min_score: int,
        max_score: int,
        label: str,
        badge: str,
        color: str,
        action: str,
        surveillance_interval: str,
        referral_level: str
    ):
        self.tier = tier
        self.min_score = min_score
        self.max_score = max_score
        self.label = label
        self.badge = badge
        self.color = color
        self.action = action
        self.surveillance_interval = surveillance_interval
        self.referral_level = referral_level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "min_score": self.min_score,
            "max_score": self.max_score if self.max_score != float('inf') else ">=16",
            "label": self.label,
            "badge": self.badge,
            "color": self.color,
            "suggested_interpretation": self.action,
            "surveillance_interval": self.surveillance_interval,
            "referral_level": self.referral_level
        }

RISK_CATEGORIES = {
    RiskTier.LOW: RiskCategory(
        tier=RiskTier.LOW,
        min_score=0,
        max_score=5,
        label="Low Risk",
        badge="🟢 Low",
        color="#10B981", # Emerald green
        action="Routine ANC",
        surveillance_interval="Monthly visits until 28w, every 2 weeks until 36w, weekly thereafter.",
        referral_level="Primary Health Centre (PHC) / Sub-Centre"
    ),
    RiskTier.MODERATE: RiskCategory(
        tier=RiskTier.MODERATE,
        min_score=6,
        max_score=10,
        label="Moderate Risk",
        badge="🟡 Moderate",
        color="#F59E0B", # Amber
        action="Additional counselling and closer monitoring",
        surveillance_interval="Every 2 weeks until 32w, weekly thereafter with focused danger sign reviews.",
        referral_level="Community Health Centre (CHC) / 24x7 Delivery Facility"
    ),
    RiskTier.HIGH: RiskCategory(
        tier=RiskTier.HIGH,
        min_score=11,
        max_score=15,
        label="High Risk",
        badge="🟠 High",
        color="#F97316", # Orange
        action="Medical review and enhanced ANC monitoring",
        surveillance_interval="Weekly / Bi-weekly specialist review, continuous fetal-maternal surveillance.",
        referral_level="Sub-District Hospital (SDH) / First Referral Unit (FRU) with Obstetrician"
    ),
    RiskTier.VERY_HIGH: RiskCategory(
        tier=RiskTier.VERY_HIGH,
        min_score=16,
        max_score=float('inf'),
        label="Very High Risk",
        badge="🔴 Very high",
        color="#EF4444", # Red
        action="Prompt specialist assessment/referral as clinically indicated",
        surveillance_interval="Immediate admission or urgent specialist referral; continuous close monitoring.",
        referral_level="Tertiary Care Hospital / Medical College with 24x7 Obstetrician, Blood Bank & NICU"
    )
}

# The 26 Clinical Risk Factor Definitions and Score Weights from Table 0 in scoring.docx
RISK_FACTORS_CONFIG = [
    {
        "id": "age_teen",
        "factor": "Age (<20 years)",
        "domain": "Demographic",
        "definition": "<20 years",
        "score": 1,
        "description": "Adolescent pregnancy with increased risks of preeclampsia, cephalopelvic disproportion, and LBW."
    },
    {
        "id": "age_advanced",
        "factor": "Age (>=35 years)",
        "domain": "Demographic",
        "definition": ">=35 years",
        "score": 1,
        "description": "Advanced maternal age associated with chromosomal anomalies, gestational hypertension, and diabetes."
    },
    {
        "id": "prev_preterm",
        "factor": "Previous preterm birth",
        "domain": "Obstetric History",
        "definition": "Yes",
        "score": 3,
        "description": "Strongest historical predictor of recurrent preterm labor."
    },
    {
        "id": "prev_lbw",
        "factor": "Previous low-birth-weight baby",
        "domain": "Obstetric History",
        "definition": "Yes",
        "score": 2,
        "description": "Previous history of baby <2.5 kg at birth, indicating risk of recurrence or fetal growth restriction."
    },
    {
        "id": "prev_stillbirth",
        "factor": "Previous stillbirth",
        "domain": "Obstetric History",
        "definition": "Yes",
        "score": 1,
        "description": "Previous intrauterine fetal demise requiring close serial fetal surveillance."
    },
    {
        "id": "prev_lscs",
        "factor": "Previous LSCS",
        "domain": "Obstetric History",
        "definition": "Yes",
        "score": 1,
        "description": "Previous lower segment Caesarean section (uterine scar management and delivery planning)."
    },
    {
        "id": "multiple_pregnancy",
        "factor": "Multiple pregnancy",
        "domain": "Current Pregnancy",
        "definition": "Yes",
        "score": 3,
        "description": "Twins/triplets with high propensity for preterm birth, malpresentation, preeclampsia, and PPH."
    },
    {
        "id": "prev_abortions",
        "factor": "Previous abortions",
        "domain": "Obstetric History",
        "definition": ">=2",
        "score": 1,
        "description": "Recurrent pregnancy loss (>=2 spontaneous or induced abortions)."
    },
    {
        "id": "birth_interval",
        "factor": "Birth interval",
        "domain": "Obstetric History",
        "definition": "<2 years",
        "score": 1,
        "description": "Short interpregnancy interval (<24 months) linked to maternal nutritional depletion and adverse outcomes."
    },
    {
        "id": "ga_assessment",
        "factor": "Gestational age at assessment",
        "domain": "Clinical Assessment",
        "definition": "<28 weeks",
        "score": 1,
        "description": "Early assessment window critical for early screening, prophylaxis, and timely intervention."
    },
    {
        "id": "hb_severe",
        "factor": "Haemoglobin (<8 g/dL)",
        "domain": "Laboratory",
        "definition": "<8 g/dL",
        "score": 3,
        "description": "Severe anaemia with heightened risk of cardiac compromise, fetal hypoxia, and postpartum hemorrhage."
    },
    {
        "id": "hb_moderate",
        "factor": "Haemoglobin (8-9.9 g/dL)",
        "domain": "Laboratory",
        "definition": "8–9.9 g/dL",
        "score": 2,
        "description": "Moderate anaemia necessitating therapeutic oral/injectable iron supplementation."
    },
    {
        "id": "bp_sys_stage1",
        "factor": "Systolic BP (130-139 mmHg)",
        "domain": "Vitals",
        "definition": "130–139 mmHg",
        "score": 1,
        "description": "Elevated / Stage 1 systolic blood pressure indicating pre-hypertensive vigilance."
    },
    {
        "id": "bp_sys_stage2",
        "factor": "Systolic BP (>=140 mmHg)",
        "domain": "Vitals",
        "definition": ">=140 mmHg",
        "score": 3,
        "description": "Hypertension threshold defining gestational hypertension or preeclampsia."
    },
    {
        "id": "bp_dia_stage1",
        "factor": "Diastolic BP (80-89 mmHg)",
        "domain": "Vitals",
        "definition": "80–89 mmHg",
        "score": 1,
        "description": "Elevated diastolic pressure."
    },
    {
        "id": "bp_dia_stage2",
        "factor": "Diastolic BP (>=90 mmHg)",
        "domain": "Vitals",
        "definition": ">=90 mmHg",
        "score": 3,
        "description": "Hypertensive diastolic pressure threshold requiring antihypertensive and proteinuria evaluation."
    },
    {
        "id": "urine_protein",
        "factor": "Urine protein",
        "domain": "Laboratory",
        "definition": "Positive",
        "score": 3,
        "description": "Proteinuria (>=1+ on dipstick or >=300mg/24h), hallmark diagnostic criterion for preeclampsia."
    },
    {
        "id": "edema",
        "factor": "Edema",
        "domain": "Clinical Signs",
        "definition": "Present",
        "score": 1,
        "description": "Pathological pedal or facial/generalized swelling."
    },
    {
        "id": "usg_abnormal",
        "factor": "Abnormal USG",
        "domain": "Imaging",
        "definition": "Yes",
        "score": 2,
        "description": "Ultrasound findings showing placenta previa, oligohydramnios, polyhydramnios, IUGR, or anomalies."
    },
    {
        "id": "medical_condition",
        "factor": "Medical condition",
        "domain": "Medical History",
        "definition": "Yes",
        "score": 2,
        "description": "Pre-existing comorbidities such as diabetes, chronic hypertension, heart disease, thyroid, or renal disorder."
    },
    {
        "id": "inadequate_weight_gain",
        "factor": "Inadequate weight gain",
        "domain": "Nutritional",
        "definition": "Yes",
        "score": 2,
        "description": "Sub-optimal gestational weight gain (<1 kg/month in 2nd/3rd trimester or <=2 kg total), linked to IUGR/LBW."
    },
    {
        "id": "anc_visits",
        "factor": "ANC visits (<4 visits)",
        "domain": "Health System",
        "definition": "<4 visits",
        "score": 2,
        "description": "Substandard antenatal care attendance (<4 visits), missing critical danger sign screening windows."
    },
    {
        "id": "ifa_noncompliance",
        "factor": "IFA (Iron Folic Acid)",
        "domain": "Nutritional / Prophylaxis",
        "definition": "Not taking",
        "score": 1,
        "description": "Non-compliance or lack of Iron and Folic Acid tablets."
    },
    {
        "id": "tt_incomplete",
        "factor": "TT (Tetanus Toxoid)",
        "domain": "Immunization",
        "definition": "Not received/incomplete",
        "score": 1,
        "description": "Unimmunized or incomplete maternal tetanus toxoid vaccination."
    },
    {
        "id": "poor_diet",
        "factor": "Poor diet",
        "domain": "Nutritional",
        "definition": "Yes",
        "score": 1,
        "description": "Inadequate caloric or micronutrient intake, lack of dietary diversity."
    },
    {
        "id": "tobacco_use",
        "factor": "Tobacco use",
        "domain": "Behavioral",
        "definition": "Yes",
        "score": 2,
        "description": "Tobacco smoking or chewing (smokeless), linked to placental insufficiency, LBW, and stillbirth."
    },
    {
        "id": "long_distance",
        "factor": "Long distance to facility",
        "domain": "Socio-Environmental",
        "definition": "Yes",
        "score": 1,
        "description": "Residing >10 km from the nearest emergency obstetric care center."
    },
    {
        "id": "transport_difficulty",
        "factor": "Transport difficulty",
        "domain": "Socio-Environmental",
        "definition": "Yes",
        "score": 1,
        "description": "Lack of reliable emergency vehicular transportation for obstetric emergencies."
    },
    {
        "id": "inadequate_support",
        "factor": "Inadequate support",
        "domain": "Social",
        "definition": "Yes",
        "score": 1,
        "description": "Lack of family/social caregiver assistance during pregnancy and postpartum."
    },
    {
        "id": "delay_in_care",
        "factor": "Delay in seeking/receiving care",
        "domain": "Health System / Behavioral",
        "definition": "Yes",
        "score": 1,
        "description": "Encountered delays (decision-making, reaching facility, or receiving timely institutional care)."
    }
]
