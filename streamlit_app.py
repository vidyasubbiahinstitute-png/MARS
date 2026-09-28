"""MAORS - Maternal Adverse Outcome Risk Score: Streamlit front end.

    streamlit run streamlit_app.py

Uses the same `maors` engine, validator, ML predictor and pipeline as the FastAPI app
(app.py), so scores, tiers, alerts and probabilities are identical. Nothing is scored here.
"""
from __future__ import annotations

import io
import os
import sys
from dataclasses import asdict

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from maors import (MAORSEngine, MAORSMLPredictor, MAORSPipeline, PatientData,  # noqa: E402
                   RISK_CATEGORIES, RISK_FACTORS_CONFIG)
from maors.validator import PatientValidator  # noqa: E402

st.set_page_config(page_title="MAORS · Maternal Risk Score", page_icon="🤰", layout="wide")


# ------------------------------------------------------------------ engines & data
@st.cache_resource(show_spinner="Loading scoring engine and ML models…")
def engines():
    eng = MAORSEngine()
    ml = MAORSMLPredictor()
    return eng, ml, MAORSPipeline(engine=eng, ml_predictor=ml)


ENGINE, ML, PIPELINE = engines()
DATASET_PATH = os.path.join(BASE_DIR, "maternal_ai_dataset.xlsx")


@st.cache_data(show_spinner="Scoring the sample cohort…")
def sample_cohort():
    if not os.path.exists(DATASET_PATH):
        return None, None
    return PIPELINE.process_dataset(pd.read_excel(DATASET_PATH), include_ml=True)


def assess(raw: dict):
    patient, errors, warnings = PatientValidator.validate_and_sanitize(raw)
    if patient is None:
        return None, errors, warnings
    res = ENGINE.evaluate(patient, include_ml=True, ml_predictor=ML).to_dict()
    return res, errors, warnings


def whatif(baseline: dict, iv: dict) -> dict:
    """Same intervention logic as POST /api/whatif in app.py."""
    sim = baseline.copy()
    if "hb" in iv:
        sim["hb"] = iv["hb"]
    if "bp_sys" in iv:
        sim["bp_sys"] = iv["bp_sys"]
        sim["bp_dia"] = iv.get("bp_dia", min(sim.get("bp_dia", 80), int(iv["bp_sys"] * 0.65)))
        if sim["bp_sys"] < 140:
            sim["urine_protein"] = False
    for k in ("diet", "transport", "ifa", "tobacco"):
        if k in iv:
            sim[k] = bool(iv[k])
    if "anc_visits" in iv:
        sim["anc_visits"] = int(iv["anc_visits"])
    return sim


def to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="MAORS_Scored_Cohort")
    return buf.getvalue()


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


# ------------------------------------------------------------------ rendering
def tier_banner(r: dict) -> None:
    c = r["color_hex"]
    st.markdown(
        f"""<div style="border-left:8px solid {c};padding:14px 18px;border-radius:8px;
        background:{c}1A;margin-bottom:8px">
        <div style="font-size:1.6rem;font-weight:700;color:{c}">{r['badge']} · {r['risk_category_label']}</div>
        <div style="font-size:1.1rem">MAORS score <b>{r['total_score']}</b> — {r['suggested_interpretation']}</div>
        <div style="opacity:.85;margin-top:4px"><b>Referral:</b> {r['referral_level']}<br>
        <b>Surveillance:</b> {r['surveillance_interval']}</div></div>""",
        unsafe_allow_html=True)


def show_result(r: dict, warnings) -> None:
    tier_banner(r)
    for w in warnings or []:
        st.warning(w)
    ml = r.get("ml_prediction") or {}
    if ml:
        c = st.columns(4)
        c[0].metric("Preterm birth", pct(ml.get("preterm_risk_prob")))
        c[1].metric("Low birth weight", pct(ml.get("lbw_risk_prob")))
        c[2].metric("Preeclampsia", pct(ml.get("preeclampsia_risk_prob")))
        c[3].metric("Any adverse outcome", pct(ml.get("composite_adverse_prob")))
        st.caption(f"ML model {ml.get('model_version')} · confidence {ml.get('confidence_tier')} · "
                   "top drivers: " + ", ".join(ml.get("top_driving_factors") or ["—"]))
    else:
        st.info("ML models not loaded — showing rule-based score only.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Clinical alerts**")
        if r["clinical_alerts"]:
            for a in r["clinical_alerts"]:
                if isinstance(a, dict):
                    box = st.error if str(a.get("severity", "")).upper() in ("CRITICAL", "EMERGENCY") \
                        else st.warning
                    box(f"**{a.get('title', 'Alert')}** ({a.get('severity', '')}, "
                        f"{a.get('category', '')})\n\n{a.get('message', '')}\n\n"
                        f"*Action:* {a.get('action_required', '')}")
                else:
                    st.error(str(a))
        else:
            st.success("No clinical alerts.")
        st.markdown("**Action plan**")
        st.markdown("\n".join(f"- {a}" for a in r["action_plan"]))
    with right:
        st.markdown("**Vitals summary**")
        vs = r["vitals_summary"]
        st.dataframe(pd.DataFrame({"measure": list(vs), "value": [str(v) for v in vs.values()]}),
                     hide_index=True, width="stretch")
    st.markdown("**Score breakdown (triggered factors)**")
    bd = pd.DataFrame(r["risk_breakdown"])
    if not bd.empty:
        st.dataframe(bd[["domain", "factor_name", "observed_value", "points_awarded",
                         "max_possible_points", "clinical_note"]],
                     hide_index=True, width="stretch")


# ------------------------------------------------------------------ patient form
def patient_form(prefix: str = "p") -> dict | None:
    d = PatientData()
    with st.form(f"{prefix}_form"):
        st.markdown("##### Demographics")
        c = st.columns(4)
        pid = c[0].text_input("Patient ID", d.id)
        age = c[1].number_input("Age (years)", 10, 60, d.age)
        height = c[2].number_input("Height (cm)", 100.0, 220.0, d.height)
        weight = c[3].number_input("Weight (kg)", 25.0, 200.0, d.weight)

        st.markdown("##### Obstetric history")
        c = st.columns(4)
        gravida = c[0].number_input("Gravida", 1, 20, d.gravida)
        ga = c[1].number_input("Gestational age (weeks)", 4, 45, d.ga_weeks)
        abortions = c[2].number_input("Previous abortions", 0, 20, d.abortions)
        interval = c[3].number_input("Birth interval (years)", 0.0, 25.0, d.birth_interval, 0.5)
        c = st.columns(4)
        prev_sb = c[0].checkbox("Previous stillbirth", d.prev_stillbirth)
        prev_pt = c[1].checkbox("Previous preterm", d.prev_preterm)
        prev_lbw = c[2].checkbox("Previous LBW baby", d.prev_lbw)
        prev_lscs = c[3].checkbox("Previous C-section", d.prev_lscs)
        c = st.columns(4)
        multiple = c[0].checkbox("Multiple pregnancy", d.multiple)
        position = c[1].selectbox("Baby position", ["Head", "NotHead"],
                                  format_func=lambda x: "Cephalic" if x == "Head" else "Breech / transverse")

        st.markdown("##### Vitals and investigations")
        c = st.columns(4)
        hb = c[0].number_input("Haemoglobin (g/dL)", 3.0, 20.0, d.hb, 0.1)
        sys_ = c[1].number_input("Systolic BP (mmHg)", 60, 260, d.bp_sys)
        dia = c[2].number_input("Diastolic BP (mmHg)", 30, 160, d.bp_dia)
        sugar = c[3].number_input("Blood sugar (mg/dL)", 40.0, 500.0, d.sugar)
        c = st.columns(4)
        wg = c[0].number_input("Weight gain to date (kg)", -15.0, 40.0, d.weight_gain, 0.5)
        fh = c[1].number_input("Fundal height (cm)", 10.0, 50.0, d.fundal_height, 0.5)
        c = st.columns(4)
        edema = c[0].checkbox("Edema", d.edema)
        fm = c[1].checkbox("Fetal movement felt", d.fetal_movement)
        up = c[2].checkbox("Urine protein positive", d.urine_protein)
        usg = c[3].checkbox("Abnormal ultrasound", d.usg_abnormal)
        medcond = st.checkbox("Pre-existing medical condition (HTN, DM, cardiac…)", d.medical_condition)

        st.markdown("##### Antenatal care and nutrition")
        c = st.columns(5)
        anc = c[0].number_input("ANC visits", 0, 25, d.anc_visits)
        tt = c[1].checkbox("TT immunised", d.tt)
        ifa = c[2].checkbox("Taking IFA", d.ifa)
        diet = c[3].checkbox("Adequate diet", d.diet)
        tobacco = c[4].checkbox("Tobacco use", d.tobacco)

        st.markdown("##### Socio-environmental")
        c = st.columns(4)
        dist = c[0].number_input("Distance to facility (km)", 0.0, 200.0, d.distance, 0.5)
        transport = c[1].checkbox("Transport difficulty", d.transport)
        support = c[2].checkbox("Family support", d.support)
        delay = c[3].checkbox("Delay in seeking care", d.delay)

        go = st.form_submit_button("Assess risk", type="primary")
    if not go:
        return None
    return dict(id=pid, age=int(age), height=height, weight=weight, gravida=int(gravida),
                ga_weeks=int(ga), prev_stillbirth=prev_sb, prev_preterm=prev_pt,
                prev_lbw=prev_lbw, prev_lscs=prev_lscs, abortions=int(abortions),
                birth_interval=interval, hb=hb, bp_sys=int(sys_), bp_dia=int(dia), sugar=sugar,
                weight_gain=wg, edema=edema, fetal_movement=fm, fundal_height=fh,
                multiple=multiple, baby_position=position, urine_protein=up, usg_abnormal=usg,
                medical_condition=medcond, anc_visits=int(anc), tt=tt, ifa=ifa, diet=diet,
                tobacco=tobacco, distance=dist, transport=transport, support=support, delay=delay)


# ------------------------------------------------------------------ page
st.title("🤰 MAORS — Maternal Adverse Outcome Risk Score")
st.caption("Rule-based clinical risk score with gradient-boosting estimates for preterm birth, "
           "low birth weight and preeclampsia.")
st.warning("Research prototype for decision support only. It is not a validated medical device "
           "and must not replace clinical judgement.", icon="⚠️")

tabs = st.tabs(["Patient assessment", "What-if simulator", "Cohort screening", "Scoring rules"])

# --- assessment
with tabs[0]:
    raw = patient_form()
    if raw is not None:
        res, errs, warns = assess(raw)
        st.session_state["assess"] = (raw, res, errs, warns)
    if "assess" in st.session_state:
        raw, res, errs, warns = st.session_state["assess"]
        st.divider()
        if res is None:
            st.error("Could not assess:\n\n" + "\n".join(f"- {e}" for e in errs))
        else:
            show_result(res, warns)

# --- what-if
with tabs[1]:
    base = st.session_state.get("assess")
    if not base or base[1] is None:
        st.info("Assess a patient on the first tab, then come back here to model interventions.")
    else:
        raw, bres = base[0], base[1]
        st.markdown(f"Baseline: **{raw['id']}** — score **{bres['total_score']}** "
                    f"({bres['risk_category_label']})")
        with st.form("wf"):
            c = st.columns(3)
            iv = {}
            if c[0].checkbox("Correct anaemia"):
                iv["hb"] = c[0].slider("Target Hb (g/dL)", 7.0, 14.0, max(11.0, raw["hb"]), 0.1)
            if c[1].checkbox("Control blood pressure"):
                iv["bp_sys"] = c[1].slider("Target systolic (mmHg)", 90, 160, min(130, raw["bp_sys"]))
            if c[2].checkbox("Increase ANC visits"):
                iv["anc_visits"] = c[2].slider("ANC visits", 0, 12, max(4, raw["anc_visits"]))
            c = st.columns(4)
            if c[0].checkbox("Nutritional support (adequate diet)"):
                iv["diet"] = True
            if c[1].checkbox("Start IFA"):
                iv["ifa"] = True
            if c[2].checkbox("Stop tobacco"):
                iv["tobacco"] = False
            if c[3].checkbox("Arrange transport"):
                iv["transport"] = False
            go = st.form_submit_button("Simulate", type="primary")
        if go:
            if not iv:
                st.warning("Pick at least one intervention.")
            else:
                sres, serrs, _ = assess(whatif(raw, iv))
                if sres is None:
                    st.error("\n".join(serrs))
                else:
                    delta = sres["total_score"] - bres["total_score"]
                    c = st.columns(5)
                    c[0].metric("MAORS score", sres["total_score"], delta, delta_color="inverse")
                    bm, sm = bres.get("ml_prediction") or {}, sres.get("ml_prediction") or {}
                    for col, (lab, k) in zip(c[1:], [("Preterm", "preterm_risk_prob"),
                                                     ("LBW", "lbw_risk_prob"),
                                                     ("Preeclampsia", "preeclampsia_risk_prob"),
                                                     ("Any adverse", "composite_adverse_prob")]):
                        if k in sm:
                            col.metric(lab, pct(sm[k]),
                                       f"{(sm[k] - bm.get(k, 0)) * 100:+.1f} pts",
                                       delta_color="inverse")
                    if sres["risk_tier"] != bres["risk_tier"]:
                        st.success(f"Tier changes: {bres['risk_category_label']} → "
                                   f"{sres['risk_category_label']}")
                    else:
                        st.info(f"Tier unchanged: {sres['risk_category_label']}")
                    with st.expander("Full simulated assessment"):
                        show_result(sres, [])

# --- cohort
with tabs[2]:
    src = st.radio("Data source", ["Sample dataset (500 patients)", "Upload Excel / CSV"],
                   horizontal=True)
    scored, summary = None, None
    if src.startswith("Sample"):
        scored, summary = sample_cohort()
        if scored is None:
            st.error("maternal_ai_dataset.xlsx not found.")
    else:
        up = st.file_uploader("Cohort file", type=["xlsx", "xls", "csv"],
                              help="Same column names as the sample dataset (ID, Age, Hb, BP_sys…).")
        if up is not None:
            try:
                df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
                with st.spinner(f"Scoring {len(df)} patients…"):
                    scored, summary = PIPELINE.process_dataset(df, include_ml=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to process file: {e}")
    if scored is not None:
        c = st.columns(4)
        c[0].metric("Screened", summary["total_screened"])
        c[1].metric("Mean score", summary["mean_score"])
        c[2].metric("Score range", f"{summary['min_score']}–{summary['max_score']}")
        c[3].metric("Clinical alerts", summary["total_clinical_alerts_flagged"])
        order = [t.value for t in RISK_CATEGORIES]
        dist = pd.DataFrame({"tier": order,
                             "patients": [summary["tier_distribution"].get(t, 0) for t in order]})
        a, b = st.columns([2, 3])
        with a:
            st.markdown("**Risk tier distribution**")
            st.bar_chart(dist, x="tier", y="patients")
        with b:
            strat = summary.get("outcome_stratification") or {}
            if strat:
                st.markdown("**Observed outcomes by tier (%)**")
                st.dataframe(pd.DataFrame(strat).T.reindex([t for t in order if t in strat]),
                             width="stretch")
        tiers = st.multiselect("Filter tiers", order, default=[t for t in order
                                                                if t in summary["tier_distribution"]])
        view = scored[scored["MAORS_Risk_Tier"].isin(tiers)] if tiers else scored
        front = ["ID", "MAORS_Score", "MAORS_Risk_Label", "MAORS_Alert_Count", "ML_Prob_Preterm",
                 "ML_Prob_LBW", "ML_Prob_Preeclampsia", "ML_Prob_Composite_Adverse"]
        front = [c for c in front if c in view.columns]
        st.dataframe(view[front + [c for c in view.columns if c not in front]],
                     hide_index=True, width="stretch", height=420)
        c = st.columns(2)
        c[0].download_button("Download scored cohort (.xlsx)", to_excel(scored),
                             "MAORS_Scored_Dataset.xlsx")
        c[1].download_button("Download scored cohort (.csv)", scored.to_csv(index=False),
                             "MAORS_Scored_Dataset.csv")

# --- rules
with tabs[3]:
    st.markdown("**Risk tiers**")
    st.dataframe(pd.DataFrame([c.to_dict() for c in RISK_CATEGORIES.values()])
                 [["badge", "min_score", "max_score", "suggested_interpretation", "referral_level",
                   "surveillance_interval"]].astype(str), hide_index=True, width="stretch")
    st.markdown(f"**Risk factors ({len(RISK_FACTORS_CONFIG)})**")
    st.dataframe(pd.DataFrame(RISK_FACTORS_CONFIG)[["domain", "factor", "definition", "score",
                                                    "description"]],
                 hide_index=True, width="stretch")
