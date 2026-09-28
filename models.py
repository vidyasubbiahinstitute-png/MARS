"""
Data models and schemas for MAORS risk assessment system.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from .constants import RiskTier, RISK_CATEGORIES

@dataclass
class PatientData:
    """Input demographic, obstetric, clinical, laboratory, and socio-environmental parameters."""
    id: Optional[str] = "P001"
    age: int = 25
    height: Optional[float] = 155.0          # in cm
    weight: Optional[float] = 55.0           # in kg
    gravida: Optional[int] = 1              # total number of pregnancies
    ga_weeks: int = 24                      # gestational age in completed weeks
    prev_stillbirth: bool = False           # history of previous stillbirth
    prev_preterm: bool = False              # history of previous preterm delivery (<37 weeks)
    prev_lbw: bool = False                  # history of low-birth-weight baby (<2.5 kg)
    prev_lscs: bool = False                 # history of lower segment Caesarean section
    abortions: int = 0                      # number of previous abortions (miscarriages/terminations)
    birth_interval: Optional[float] = 3.0   # interpregnancy interval in years
    hb: float = 11.0                        # haemoglobin in g/dL
    bp_sys: int = 120                       # systolic blood pressure in mmHg
    bp_dia: int = 80                        # diastolic blood pressure in mmHg
    sugar: Optional[float] = 100.0          # blood sugar in mg/dL (random or fasting)
    weight_gain: Optional[float] = 5.0      # total gestational weight gain to date in kg
    edema: bool = False                     # presence of clinical edema (pedal / facial)
    fetal_movement: bool = True             # active fetal movement felt (for GA >= 20w)
    fundal_height: Optional[float] = 24.0   # symphysis-fundal height in cm
    multiple: bool = False                  # multiple pregnancy (twins/triplets)
    baby_position: Optional[str] = "Head"   # 'Head' (cephalic) or 'NotHead' (breech/transverse)
    urine_protein: bool = False             # urine albumin / dipstick positive
    usg_abnormal: bool = False              # abnormal ultrasound finding
    medical_condition: bool = False         # pre-existing medical disorder (HTN, DM, cardiac, etc.)
    anc_visits: int = 4                     # number of completed antenatal checkups to date
    tt: bool = True                         # received tetanus toxoid immunizations
    ifa: bool = True                        # adhering to iron-folic acid supplementation
    diet: bool = True                       # adequate nutritional diet (True = adequate, False = poor)
    tobacco: bool = False                   # history of tobacco consumption (smoke/smokeless)
    distance: Optional[float] = 5.0         # distance from home to nearest obstetric facility in km
    transport: bool = False                 # transport difficulty present (True = difficulty, False = accessible)
    support: bool = True                    # family/social support present (True = adequate, False = inadequate)
    delay: bool = False                     # delay in seeking or receiving care reported

    def get_bmi(self) -> Optional[float]:
        if self.height and self.weight and self.height > 0:
            height_m = self.height / 100.0
            return round(self.weight / (height_m ** 2), 2)
        return None

    def get_map(self) -> float:
        """Mean Arterial Pressure = (2*Diastolic + Systolic) / 3"""
        return round((2.0 * self.bp_dia + self.bp_sys) / 3.0, 2)

    def get_pulse_pressure(self) -> int:
        return self.bp_sys - self.bp_dia

    def get_fundal_lag(self) -> Optional[float]:
        if self.fundal_height is not None:
            return round(self.ga_weeks - self.fundal_height, 2)
        return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientData":
        """Helper to instantiate PatientData from dataset row or dictionary with flexible key naming."""
        def parse_bool(val, default=False) -> bool:
            if val is None:
                return default
            if isinstance(val, (bool, np_bool := getattr(__import__('numpy'), 'bool_', bool))):
                return bool(val)
            if isinstance(val, (int, float)):
                return val > 0
            if isinstance(val, str):
                return val.strip().lower() in ['yes', 'true', '1', 'positive', 'present', 't', 'y']
            return default

        # Map dataset columns to PatientData fields
        id_val = str(data.get('ID', data.get('id', 'P001')))
        age = int(data.get('Age', data.get('age', 25)))
        height = float(data.get('Height', data.get('height', 155.0))) if 'Height' in data or 'height' in data else None
        weight = float(data.get('Weight', data.get('weight', 55.0))) if 'Weight' in data or 'weight' in data else None
        gravida = int(data.get('Gravida', data.get('gravida', 1))) if 'Gravida' in data or 'gravida' in data else None
        ga_weeks = int(data.get('GA_weeks', data.get('ga_weeks', data.get('GA', 24))))
        
        prev_stillbirth = parse_bool(data.get('Prev_Stillbirth', data.get('prev_stillbirth', 0)))
        prev_preterm = parse_bool(data.get('Prev_Preterm', data.get('prev_preterm', 0)))
        prev_lbw = parse_bool(data.get('Prev_LBW', data.get('prev_lbw', 0)))
        prev_lscs = parse_bool(data.get('Prev_LSCS', data.get('prev_lscs', 0)))
        abortions = int(data.get('Abortions', data.get('abortions', 0)))
        birth_interval = float(data.get('Birth_Interval', data.get('birth_interval', 3.0))) if ('Birth_Interval' in data or 'birth_interval' in data) else None
        
        hb = float(data.get('Hb', data.get('hb', 11.0)))
        bp_sys = int(data.get('BP_sys', data.get('bp_sys', data.get('Systolic', 120))))
        bp_dia = int(data.get('BP_dia', data.get('bp_dia', data.get('Diastolic', 80))))
        sugar = float(data.get('Sugar', data.get('sugar', 100.0))) if ('Sugar' in data or 'sugar' in data) else None
        weight_gain = float(data.get('Weight_Gain', data.get('weight_gain', 5.0))) if ('Weight_Gain' in data or 'weight_gain' in data) else None
        edema = parse_bool(data.get('Edema', data.get('edema', 0)))
        
        fetal_movement = parse_bool(data.get('Fetal_Movement', data.get('fetal_movement', 1)), default=True)
        fundal_height = float(data.get('Fundal_Height', data.get('fundal_height', ga_weeks))) if ('Fundal_Height' in data or 'fundal_height' in data) else None
        multiple = parse_bool(data.get('Multiple', data.get('multiple', 0)))
        baby_position = str(data.get('Baby_Position', data.get('baby_position', 'Head')))
        urine_protein = parse_bool(data.get('Urine_Protein', data.get('urine_protein', 0)))
        usg_abnormal = parse_bool(data.get('USG_Abnormal', data.get('usg_abnormal', 0)))
        medical_condition = parse_bool(data.get('Medical_Condition', data.get('medical_condition', 0)))
        
        anc_visits = int(data.get('ANC_Visits', data.get('anc_visits', 4)))
        tt = parse_bool(data.get('TT', data.get('tt', 1)), default=True)
        ifa = parse_bool(data.get('IFA', data.get('ifa', 1)), default=True)
        
        # In dataset: Diet=1 is adequate, Diet=0 is poor.
        diet_raw = data.get('Diet', data.get('diet', 1))
        diet = parse_bool(diet_raw, default=True)
        
        tobacco = parse_bool(data.get('Tobacco', data.get('tobacco', 0)))
        distance = float(data.get('Distance', data.get('distance', 5.0))) if ('Distance' in data or 'distance' in data) else None
        transport = parse_bool(data.get('Transport', data.get('transport', 0)))
        
        # Support: 1=adequate support, 0=inadequate support
        support_raw = data.get('Support', data.get('support', 1))
        support = parse_bool(support_raw, default=True)
        
        delay = parse_bool(data.get('Delay', data.get('delay', 0)))

        return cls(
            id=id_val,
            age=age,
            height=height,
            weight=weight,
            gravida=gravida,
            ga_weeks=ga_weeks,
            prev_stillbirth=prev_stillbirth,
            prev_preterm=prev_preterm,
            prev_lbw=prev_lbw,
            prev_lscs=prev_lscs,
            abortions=abortions,
            birth_interval=birth_interval,
            hb=hb,
            bp_sys=bp_sys,
            bp_dia=bp_dia,
            sugar=sugar,
            weight_gain=weight_gain,
            edema=edema,
            fetal_movement=fetal_movement,
            fundal_height=fundal_height,
            multiple=multiple,
            baby_position=baby_position,
            urine_protein=urine_protein,
            usg_abnormal=usg_abnormal,
            medical_condition=medical_condition,
            anc_visits=anc_visits,
            tt=tt,
            ifa=ifa,
            diet=diet,
            tobacco=tobacco,
            distance=distance,
            transport=transport,
            support=support,
            delay=delay
        )


@dataclass
class RiskBreakdownItem:
    """Detailed accounting for each individual risk factor."""
    factor_id: str
    factor_name: str
    domain: str
    definition: str
    points_awarded: int
    max_possible_points: int
    is_triggered: bool
    observed_value: Any
    clinical_note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalAlert:
    """Specific high-priority clinical warnings triggered by specific combinations of risk factors."""
    title: str
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    category: str
    message: str
    action_required: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MLPredictionResult:
    """Probabilistic predictions from the trained Machine Learning modules."""
    preterm_risk_prob: float
    lbw_risk_prob: float
    preeclampsia_risk_prob: float
    composite_adverse_prob: float
    model_version: str = "1.0.0-ensemble"
    confidence_tier: str = "High"
    top_driving_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssessmentResult:
    """Complete, end-to-end MAORS risk assessment report."""
    patient_id: str
    total_score: int
    risk_tier: str
    risk_category_label: str
    badge: str
    color_hex: str
    suggested_interpretation: str
    surveillance_interval: str
    referral_level: str
    clinical_alerts: List[ClinicalAlert] = field(default_factory=list)
    action_plan: List[str] = field(default_factory=list)
    risk_breakdown: List[RiskBreakdownItem] = field(default_factory=list)
    ml_prediction: Optional[MLPredictionResult] = None
    vitals_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "total_score": self.total_score,
            "risk_tier": self.risk_tier,
            "risk_category_label": self.risk_category_label,
            "badge": self.badge,
            "color_hex": self.color_hex,
            "suggested_interpretation": self.suggested_interpretation,
            "surveillance_interval": self.surveillance_interval,
            "referral_level": self.referral_level,
            "vitals_summary": self.vitals_summary,
            "clinical_alerts": [a.to_dict() for a in self.clinical_alerts],
            "action_plan": self.action_plan,
            "ml_prediction": self.ml_prediction.to_dict() if self.ml_prediction else None,
            "risk_breakdown": [b.to_dict() for b in self.risk_breakdown if b.is_triggered]
        }
