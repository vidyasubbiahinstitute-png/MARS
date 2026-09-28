"""
Input validation and clinical sanitization for MAORS.
"""

from typing import Dict, Any, Tuple, List, Optional
from .models import PatientData

PHYSIOLOGICAL_BOUNDS = {
    "age": (10, 60, "Maternal Age (years)"),
    "ga_weeks": (4, 45, "Gestational Age (weeks)"),
    "hb": (3.0, 20.0, "Haemoglobin (g/dL)"),
    "bp_sys": (60, 260, "Systolic Blood Pressure (mmHg)"),
    "bp_dia": (30, 160, "Diastolic Blood Pressure (mmHg)"),
    "height": (100.0, 220.0, "Height (cm)"),
    "weight": (25.0, 200.0, "Weight (kg)"),
    "sugar": (40.0, 500.0, "Blood Sugar (mg/dL)"),
    "weight_gain": (-15.0, 40.0, "Weight Gain (kg)"),
    "fundal_height": (10.0, 50.0, "Fundal Height (cm)"),
    "anc_visits": (0, 25, "ANC Visits"),
    "abortions": (0, 20, "Previous Abortions"),
    "birth_interval": (0.0, 25.0, "Birth Interval (years)"),
    "distance": (0.0, 200.0, "Distance to Facility (km)")
}

class PatientValidator:
    """Validates patient demographic, clinical, and laboratory values."""

    @staticmethod
    def validate_and_sanitize(raw_data: Dict[str, Any]) -> Tuple[Optional[PatientData], List[str], List[str]]:
        """
        Validates raw dictionary inputs.
        Returns:
            - patient: Sanitized PatientData instance (or None if critical errors)
            - errors: List of blocking error strings
            - warnings: List of non-blocking warning strings
        """
        errors = []
        warnings = []

        try:
            patient = PatientData.from_dict(raw_data)
        except Exception as e:
            errors.append(f"Data mapping error: {str(e)}")
            return None, errors, warnings

        # Bounds validation
        for field_name, (min_val, max_val, label) in PHYSIOLOGICAL_BOUNDS.items():
            val = getattr(patient, field_name, None)
            if val is not None:
                if val < min_val or val > max_val:
                    warnings.append(
                        f"Unusual {label}: {val}. Typical clinical range is between {min_val} and {max_val}."
                    )

        # Logic consistency checks
        if patient.bp_dia >= patient.bp_sys:
            errors.append(
                f"Diastolic BP ({patient.bp_dia} mmHg) cannot be greater than or equal to Systolic BP ({patient.bp_sys} mmHg)."
            )

        if patient.ga_weeks >= 24 and patient.fundal_height is not None:
            diff = abs(patient.ga_weeks - patient.fundal_height)
            if diff > 5:
                warnings.append(
                    f"Discrepancy of {diff:.1f} cm between Gestational Age ({patient.ga_weeks}w) and Fundal Height ({patient.fundal_height}cm). Consider ultrasound for FGR/Oligohydramnios or Macrosomia/Polyhydramnios."
                )

        return patient, errors, warnings
