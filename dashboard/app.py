"""Streamlit interactive demonstration dashboard for PRISM-U active unlearning auditing.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import matplotlib.pyplot as plt
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from audit.audit_gain import AdaptivePosteriorGainEstimator, PrivacyBlindActivePolicy, StaticJSGainEstimator
from audit.evidence import SurrogateEvidenceModel
from audit.fixed_policy import FixedTopKPolicy
from audit.privacy_risk import StaticQueryRisk, TranscriptPrivacyEstimator
from audit.prism_u import (
    AdaptiveGainStaticPrivacyPolicy,
    PRISMUPolicy,
    StaticPrivacyAwarePolicy,
)
from audit.random_policy import RandomQueryPolicy
from audit.stopping import SequentialStoppingEngine, StoppingConfig
from models.black_box_target import load_cached_target_bank
from utils.io import load_yaml


def load_resources():
    cfg = load_yaml(REPO_ROOT / "configs" / "stage2.yaml")
    artifact_root = REPO_ROOT / cfg["paths"]["artifact_root"]
    surrogate_cache = artifact_root / "responses" / "surrogate_responses.npz"

    evidence_model = SurrogateEvidenceModel.from_surrogate_cache(surrogate_cache, bandwidth=1.0)
    privacy_estimator = TranscriptPrivacyEstimator.from_evidence_model(evidence_model, fast_mode=True)
    gain_estimator = AdaptivePosteriorGainEstimator(evidence_model)
    static_gain = StaticJSGainEstimator(evidence_model)
    static_risk = privacy_estimator.static_risk
    fixed_policy = FixedTopKPolicy.from_surrogate_cache(surrogate_cache, metric="js_mean")

    return {
        "cfg": cfg,
        "artifact_root": artifact_root,
        "evidence_model": evidence_model,
        "privacy_estimator": privacy_estimator,
        "gain_estimator": gain_estimator,
        "static_gain": static_gain,
        "static_risk": static_risk,
        "fixed_policy": fixed_policy,
    }


def main():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Install via: pip install streamlit")
        return

    st.set_page_config(
        page_title="PRISM-U Auditing Dashboard",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ PRISM-U: Privacy-Budgeted Active Auditing")
    st.markdown(
        "**Sequential Black-Box Compliance Verification under Retained-Membership Privacy Constraints**"
    )

    with st.spinner("Loading surrogate reference models and evidence engine..."):
        resources = load_resources()

    # Sidebar: Audit Configuration
    st.sidebar.header("Audit Configuration")

    target_bank_type = st.sidebar.selectbox(
        "Target Bank",
        options=["validation", "surrogate", "final"],
        index=0,
        help="Select model bank to audit.",
    )

    cache_file = resources["artifact_root"] / "responses" / f"{target_bank_type}_responses.npz"
    targets = load_cached_target_bank(cache_file, query_budget=160, anonymize_ids=False)

    target_options = [f"{i}: {t['ground_truth']['model_id']} ({t['ground_truth']['hypothesis']})" for i, t in enumerate(targets)]
    selected_target_idx = st.sidebar.selectbox("Select Target Model", options=range(len(targets)), format_func=lambda i: target_options[i])
    selected_item = targets[selected_target_idx]

    policy_choice = st.sidebar.selectbox(
        "Auditing Policy",
        options=[
            "Full PRISM-U (Adaptive Gain + Marginal Risk + Caps)",
            "Privacy-Blind Active (Adaptive Gain Only)",
            "Fixed Top-K (Static JS Ranking)",
            "Static Privacy-Aware (Static Gain & Risk)",
            "Random Querying",
        ],
        index=0,
    )

    bq_budget = st.sidebar.slider("Query Budget (B_Q)", min_value=5, max_value=80, value=20, step=5)
    bp_budget = st.sidebar.slider("Privacy Budget (B_P)", min_value=0.50, max_value=1.00, value=0.70, step=0.05)
    theta_pass = st.sidebar.number_input("PASS Threshold (theta_PASS)", value=2.944, step=0.5)
    theta_fail = st.sidebar.number_input("FAIL Threshold (theta_FAIL)", value=-2.944, step=0.5)

    ev_model = resources["evidence_model"]
    priv_est = resources["privacy_estimator"]
    gain_est = resources["gain_estimator"]

    # Instantiate selected policy
    target_obj = selected_item["target"]
    pool_size = target_obj.pool_size

    if "Full PRISM-U" in policy_choice:
        policy = PRISMUPolicy(
            pool_size=pool_size,
            evidence_model=ev_model,
            privacy_budget=bp_budget,
            privacy_estimator=priv_est,
            gain_estimator=gain_est,
            penalty_lambda=1.0,
            pre_filter_k=50,
        )
    elif "Privacy-Blind" in policy_choice:
        policy = PrivacyBlindActivePolicy(pool_size=pool_size, evidence_model=ev_model, gain_estimator=gain_est)
    elif "Fixed Top-K" in policy_choice:
        policy = resources["fixed_policy"]
    elif "Static Privacy" in policy_choice:
        policy = StaticPrivacyAwarePolicy(
            pool_size=pool_size,
            evidence_model=ev_model,
            penalty_lambda=1.0,
            static_gain=resources["static_gain"],
            static_risk=resources["static_risk"],
        )
    else:
        policy = RandomQueryPolicy(pool_size=pool_size, seed=42)

    stop_cfg = StoppingConfig(
        theta_pass=theta_pass,
        theta_fail=theta_fail,
        query_budget=bq_budget,
        privacy_budget=bp_budget,
    )
    engine = SequentialStoppingEngine(evidence_model=ev_model, config=stop_cfg)

    # Run audit button
    if st.button("🚀 Execute Audit", type="primary"):
        with st.spinner("Executing sequential black-box audit..."):
            result = engine.run_audit(
                target=target_obj,
                policy=policy,
                privacy_estimator=priv_est,
            )

        gt = selected_item["ground_truth"]
        is_compliant = gt["hypothesis"] == "HC"

        # Result columns
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if result.verdict == "PASS":
                st.success(f"**Verdict:** {result.verdict}")
            elif result.verdict == "FAIL":
                st.error(f"**Verdict:** {result.verdict}")
            else:
                st.warning(f"**Verdict:** {result.verdict}")
            st.caption(f"Reason: `{result.reason}`")

        with c2:
            st.metric("Queries Used", f"{result.queries_used} / {bq_budget}")

        with c3:
            st.metric("Estimated Leakage L_hat", f"{result.cumulative_leakage:.3f}", delta=f"{bp_budget - result.cumulative_leakage:+.3f} remaining")

        with c4:
            st.metric("Final LLR", f"{result.final_llr:.2f}", f"P(H_C)={result.final_p_compliant:.3f}")

        # Ground truth check
        correct = (result.verdict == "PASS" and is_compliant) or (result.verdict == "FAIL" and not is_compliant)
        if result.verdict != "INCONCLUSIVE":
            if correct:
                st.info(f"✅ Ground Truth Alignment: Correct decision for `{gt['model_id']}` ({gt['condition']}).")
            else:
                st.error(f"❌ Ground Truth Misalignment: Declared `{result.verdict}` for `{gt['model_id']}` ({gt['condition']}).")
        else:
            st.info(f"ℹ️ Audit abstained within budget bounds for `{gt['model_id']}` ({gt['condition']}).")

        # Plot sequential LLR trajectory
        st.subheader("Sequential Evidence Trajectory (LLR_t)")
        steps_data = result.steps
        if steps_data:
            steps_idx = [0] + [s.step for s in steps_data]
            llr_vals = [0.0] + [s.llr for s in steps_data]
            leak_vals = [0.50] + [s.cumulative_leakage for s in steps_data]

            fig, ax1 = plt.subplots(figsize=(10, 4), dpi=150)
            ax1.plot(steps_idx, llr_vals, marker="o", color="#1f77b4", linewidth=2, label="Evidence (LLR_t)")
            ax1.axhline(theta_pass, color="green", linestyle="--", label=f"theta_PASS ({theta_pass:.2f})")
            ax1.axhline(theta_fail, color="red", linestyle="--", label=f"theta_FAIL ({theta_fail:.2f})")
            ax1.set_xlabel("Query Step t")
            ax1.set_ylabel("Log-Likelihood Ratio (LLR_t)", color="#1f77b4")
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend(loc="upper left")

            st.pyplot(fig)

            # Transcript Table
            st.subheader("Audit Interaction Transcript")
            table_rows = [
                {
                    "Step": s.step,
                    "Query ID": s.query_id,
                    "LLR": f"{s.llr:.4f}",
                    "P(Compliant)": f"{s.p_compliant:.4f}",
                    "Cumulative Leakage L_hat": f"{s.cumulative_leakage:.4f}",
                }
                for s in steps_data
            ]
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)


if __name__ == "__main__":
    main()
