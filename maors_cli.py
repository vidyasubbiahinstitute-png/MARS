"""
MAORS Command-Line Interface (CLI) Tool.
Provides interactive assessment, batch dataset processing, clinical rules exploration, and audit verification.
"""

import sys
import os
import argparse
import json
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
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

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def display_info():
    print_header("MAORS: MATERNAL ADVERSE OUTCOME RISK SCORE - CLINICAL SPECIFICATION")
    print("“A simple clinical risk score for early identification of pregnant women at risk of adverse pregnancy outcomes.”\n")
    
    print("--- RISK CLASSIFICATION TIERS (Table 1) ---")
    print(f"{'Tier':<12} | {'Score Range':<12} | {'Badge':<12} | {'Suggested Clinical Interpretation'}")
    print("-" * 75)
    for tier, cat in RISK_CATEGORIES.items():
        score_range = f"{cat.min_score}–{cat.max_score if cat.max_score != float('inf') else '>=16'}"
        print(f"{tier.value:<12} | {score_range:<12} | {cat.badge:<12} | {cat.action}")
    
    print("\n--- CLINICAL RISK FACTORS & SCORE WEIGHTS (Table 0) ---")
    print(f"{'No.':<4} | {'Risk Factor':<34} | {'Definition':<22} | {'Score':<5} | {'Domain'}")
    print("-" * 78)
    for idx, f in enumerate(RISK_FACTORS_CONFIG, 1):
        print(f"{idx:<4} | {f['factor']:<34} | {f['definition']:<22} | +{f['score']:<4} | {f['domain']}")
    print("\n")

def run_interactive():
    print_header("MAORS INTERACTIVE PATIENT CLINICAL ASSESSMENT")
    print("Enter patient clinical values (press ENTER to accept default):\n")

    def prompt(label, default, cast_type=str):
        user_val = input(f"{label} [{default}]: ").strip()
        if not user_val:
            return default
        try:
            if cast_type == bool:
                return user_val.lower() in ['y', 'yes', '1', 'true', 't']
            return cast_type(user_val)
        except Exception:
            return default

    data = {
        "id": prompt("Patient ID", "PATIENT-001"),
        "age": prompt("Maternal Age (years)", 24, int),
        "ga_weeks": prompt("Gestational Age (weeks)", 26, int),
        "hb": prompt("Haemoglobin (g/dL)", 10.5, float),
        "bp_sys": prompt("Systolic BP (mmHg)", 120, int),
        "bp_dia": prompt("Diastolic BP (mmHg)", 80, int),
        "height": prompt("Height (cm)", 155.0, float),
        "weight": prompt("Weight (kg)", 56.0, float),
        "weight_gain": prompt("Gestational Weight Gain (kg)", 4.0, float),
        "prev_preterm": prompt("Previous Preterm Birth (y/n)", "n", bool),
        "prev_lbw": prompt("Previous LBW Baby (y/n)", "n", bool),
        "prev_stillbirth": prompt("Previous Stillbirth (y/n)", "n", bool),
        "prev_lscs": prompt("Previous LSCS (y/n)", "n", bool),
        "multiple": prompt("Multiple Pregnancy (y/n)", "n", bool),
        "abortions": prompt("Previous Abortions Count", 0, int),
        "birth_interval": prompt("Interpregnancy Birth Interval (years)", 3.0, float),
        "urine_protein": prompt("Urine Protein Positive (y/n)", "n", bool),
        "edema": prompt("Pathological Edema Present (y/n)", "n", bool),
        "usg_abnormal": prompt("Abnormal USG Finding (y/n)", "n", bool),
        "medical_condition": prompt("Pre-existing Medical Condition (y/n)", "n", bool),
        "anc_visits": prompt("Completed ANC Visits Count", 4, int),
        "ifa": prompt("Taking IFA regularly (y/n)", "y", bool),
        "tt": prompt("Received TT Immunization (y/n)", "y", bool),
        "diet": prompt("Adequate Diet (y/n)", "y", bool),
        "tobacco": prompt("Tobacco Use (y/n)", "n", bool),
        "distance": prompt("Distance to Obstetric Facility (km)", 5.0, float),
        "transport": prompt("Transport Difficulty (y/n)", "n", bool),
        "support": prompt("Adequate Family/Social Support (y/n)", "y", bool),
        "delay": prompt("Experienced Delay in Seeking Care (y/n)", "n", bool)
    }

    patient = PatientData.from_dict(data)
    engine = MAORSEngine()
    ml_predictor = MAORSMLPredictor()
    result = engine.evaluate(patient, include_ml=True, ml_predictor=ml_predictor)

    display_assessment_result(result)

def display_assessment_result(result: AssessmentResult):
    print_header(f"ASSESSMENT REPORT FOR PATIENT: {result.patient_id}")
    print(f"Total MAORS Score:          {result.total_score} points")
    print(f"Clinical Risk Tier:         {result.badge} ({result.risk_category_label})")
    print(f"Action / Interpretation:    {result.suggested_interpretation}")
    print(f"Surveillance Protocol:      {result.surveillance_interval}")
    print(f"Recommended Facility Level: {result.referral_level}")
    
    print("\n--- VITALS & ANTHROPOMETRY ---")
    for k, v in result.vitals_summary.items():
        print(f"  • {k.replace('_', ' ').title():<22}: {v}")

    print("\n--- TRIGGERED RISK FACTORS ---")
    triggered_items = [b for b in result.risk_breakdown if b.is_triggered]
    if triggered_items:
        for idx, b in enumerate(triggered_items, 1):
            print(f"  {idx:2d}. [{b.domain}] {b.factor_name} (+{b.points_awarded} pts) | Observed: {b.observed_value}")
            print(f"      -> {b.clinical_note}")
    else:
        print("  🟢 No elevated risk factors triggered. Standard healthy profile.")

    if result.clinical_alerts:
        print("\n--- ⚠️ CLINICAL ALERTS & RED FLAGS ---")
        for a in result.clinical_alerts:
            print(f"  [{a.severity}] {a.title} ({a.category})")
            print(f"    Message: {a.message}")
            print(f"    Action:  {a.action_required}")

    if result.ml_prediction:
        print("\n--- 🤖 AI PREDICTIVE RISK PROBABILITIES ---")
        ml = result.ml_prediction
        print(f"  • Preterm Birth (<37w) Risk:        {ml.preterm_risk_prob * 100:5.1f}%")
        print(f"  • Low Birth Weight (<2.5kg) Risk:   {ml.lbw_risk_prob * 100:5.1f}%")
        print(f"  • Preeclampsia Risk:                {ml.preeclampsia_risk_prob * 100:5.1f}%")
        print(f"  • Overall Composite Adverse Risk:   {ml.composite_adverse_prob * 100:5.1f}%")
        if ml.top_driving_factors:
            print(f"  • Key Model Drivers: {', '.join(ml.top_driving_factors)}")

    print("\n--- 📋 CLINICAL ACTION CHECKLIST ---")
    for idx, act in enumerate(result.action_plan, 1):
        print(f"  [{idx}] {act}")
    print("=" * 70 + "\n")

def run_batch(input_file: str, output_file: str = None):
    print_header(f"PROCESSING BATCH DATASET: {input_file}")
    pipeline = MAORSPipeline()
    scored_df, summary = pipeline.process_dataset(input_file, include_ml=True)

    print(f"Successfully processed {summary['total_screened']} patient records.")
    print(f"Score Range: {summary['min_score']} to {summary['max_score']} (Mean: {summary['mean_score']}, Median: {summary['median_score']})")
    print(f"Total High-Priority Alerts Triggered: {summary['total_clinical_alerts_flagged']}")
    
    print("\n--- RISK TIER DISTRIBUTION ---")
    for tier, count in summary['tier_distribution'].items():
        pct = summary['tier_percentages'].get(tier, 0)
        badge = RISK_CATEGORIES[RiskTier(tier)].badge
        print(f"  {badge:<14} ({tier:<10}): {count:4d} patients ({pct:5.1f}%)")

    if "outcome_stratification" in summary:
        print("\n--- CLINICAL OUTCOME VALIDATION BY RISK TIER ---")
        print(f"{'Risk Tier':<12} | {'n':<6} | {'Preterm %':<11} | {'LBW %':<9} | {'Preeclampsia %':<16} | {'Any Adverse %'}")
        print("-" * 75)
        for tier, strat in summary['outcome_stratification'].items():
            badge = RISK_CATEGORIES[RiskTier(tier)].badge
            print(f"{badge:<12} | {strat['n']:<6} | {strat.get('Outcome_Preterm', 'N/A')}%{'':<5} | {strat.get('Outcome_LBW', 'N/A')}%{'':<3} | {strat.get('Outcome_Preeclampsia', 'N/A')}%{'':<10} | {strat.get('Any_Adverse_Outcome', 'N/A')}%")

    if output_file:
        if output_file.endswith('.xlsx'):
            scored_df.to_excel(output_file, index=False)
        else:
            scored_df.to_csv(output_file, index=False)
        print(f"\nSaved scored results to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="MAORS: Maternal Adverse Outcome Risk Score System")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("info", help="Display clinical scoring guidelines and weights")
    subparsers.add_parser("score", help="Run interactive single patient risk assessment")
    
    batch_parser = subparsers.add_parser("batch", help="Batch process dataset file")
    batch_parser.add_argument("--input", "-i", default="maternal_ai_dataset.xlsx", help="Input dataset path (.xlsx or .csv)")
    batch_parser.add_argument("--output", "-o", default="maternal_scored_results.xlsx", help="Output destination path")

    audit_parser = subparsers.add_parser("audit", help="Run dataset clinical validation audit")
    audit_parser.add_argument("--file", "-f", default="maternal_ai_dataset.xlsx", help="Dataset path")

    args = parser.parse_args()

    if args.command == "info":
        display_info()
    elif args.command == "score":
        run_interactive()
    elif args.command == "batch":
        run_batch(args.input, args.output)
    elif args.command == "audit":
        run_batch(args.file, output_file=None)
    else:
        # Default to audit if no arg given
        run_batch("maternal_ai_dataset.xlsx", output_file="maternal_scored_results.xlsx")

if __name__ == "__main__":
    main()
