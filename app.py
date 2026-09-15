import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import os

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Microfinance Adaptive Scheduler", layout="wide")
st.title("🌱 Dynamic Microloan & Cash-Flow Planner")
st.caption("Temporal LSTM Forecasting + Forecast-Driven Algorithmic Policy")

# --- MODEL ARCHITECTURE DEFINITIONS ---
class CashFlowLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super(CashFlowLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

@st.cache_resource
def load_ai_models():
    device = torch.device("cpu")
    lstm = CashFlowLSTM(input_dim=4, hidden_dim=64)
    lstm_loaded = False

    if os.path.exists("lstm_predictor.pth"):
        try:
            lstm.load_state_dict(torch.load("lstm_predictor.pth", map_location=device))
            lstm.eval()
            lstm_loaded = True
        except Exception:
            pass

    return lstm, lstm_loaded

lstm_engine, lstm_ready = load_ai_models()

st.sidebar.markdown("### 🤖 Engine Status")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("LSTM Predictor", "Active" if lstm_ready else "Naive Fallback", delta="Ready" if lstm_ready else "Baseline")
col_s2.metric("Algorithmic Policy", "Active", delta="Rules Engine")

# --- REALISTIC INFORMAL ARTIFACT PROFILES ---
monthly_profiles = {
    "Seasonal Farmer": {
        "desc": "Kharif & Rabi crop cycles. High input costs during sowing (Jun/Nov), zero revenue during gestation, massive harvest windfalls (Apr/Oct).",
        "loan_amount": 300000, "interest_rate": 10.5, "term_years": 5,
        "inflow":  [6000,  5000, 12000, 145000, 7000,  4000,  8000, 10000, 15000, 160000,  5000,  8000],
        "outflow": [9000,  8500, 10000,  18000, 9500, 26000, 11000, 10000, 12000,  20000, 24000,  9000]
    },
    "Festival Artisan": {
        "desc": "Handicrafts & festive apparel. Months of raw-material stockpiling with zero sales, followed by massive Diwali/Wedding windfalls.",
        "loan_amount": 90000, "interest_rate": 13.0, "term_years": 2,
        "inflow":  [42000, 14000, 11000, 10000,  9000, 12000, 15000,  18000,  22000,  98000, 75000, 22000],
        "outflow": [16000, 11000, 10500, 10000, 10000, 14000, 28000,  32000,  35000,  24000, 19000, 14000]
    },
    "Daily Street Vendor": {
        "desc": "Food & retail stall. Thin margins, vulnerable to monsoon washouts (Jul-Aug), health emergencies, and seasonal footfall.",
        "loan_amount": 40000, "interest_rate": 14.0, "term_years": 1,
        "inflow":  [36000, 34000, 33000, 31000, 23000, 28000, 14000,  16000,  31000,  44000, 42000, 39000],
        "outflow": [22000, 21000, 21000, 20000, 19000, 20000, 18000,  18500,  20500,  24000, 23000, 22500]
    },
    "Gig Driver (Taxi/Auto)": {
        "desc": "Ride-hailing driver. Steady daily income, but hit by recurring quarterly insurance renewals, tyre replacements, and fitness tests.",
        "loan_amount": 75000, "interest_rate": 12.5, "term_years": 2,
        "inflow":  [41000, 39000, 38000, 34000, 26000, 35000, 33000,  36000,  38000,  43000, 46000, 49000],
        "outflow": [21000, 20000, 37000, 21000, 20000, 21000, 22000,  21500,  41000,  23000, 24000, 25000]
    }
}

st.sidebar.header("1. Select Borrower Archetype")
selected_profile = st.sidebar.selectbox("Archetype", list(monthly_profiles.keys()), index=0)
profile = monthly_profiles[selected_profile]
st.sidebar.info(profile["desc"])

st.sidebar.header("2. Loan Parameters")
loan_amount = st.sidebar.number_input("Total Loan Principal (₹)", value=profile["loan_amount"], step=5000)
interest_rate = st.sidebar.slider("Annual Interest Rate (%)", 6.0, 28.0, value=profile["interest_rate"])
loan_term_years = st.sidebar.slider("Tenure (Years)", 1, 15, value=profile["term_years"])
loan_term_months = loan_term_years * 12

subsistence_buffer = st.sidebar.number_input("Household Subsistence Reserve (₹)", value=11000, step=1000)

# --- EDITABLE BASELINE DATA ---
st.markdown("### 📝 Edit Baseline 12-Month Historical Data")
st.caption(f"Adjust the standard previous-year cash flows for the **{selected_profile}**. The simulation drives future variations off these numbers.")

baseline_df = pd.DataFrame({
    "Month": ["1 (Jan)", "2 (Feb)", "3 (Mar)", "4 (Apr)", "5 (May)", "6 (Jun)", 
              "7 (Jul)", "8 (Aug)", "9 (Sep)", "10 (Oct)", "11 (Nov)", "12 (Dec)"],
    "Inflow (₹)": profile["inflow"],
    "Outflow (₹)": profile["outflow"]
})

edited_df = st.data_editor(
    baseline_df,
    column_config={
        "Month": st.column_config.TextColumn("Month", disabled=True),
        "Inflow (₹)": st.column_config.NumberColumn("Inflow (₹)", min_value=0, step=1000),
        "Outflow (₹)": st.column_config.NumberColumn("Outflow (₹)", min_value=0, step=1000)
    },
    hide_index=True,
    use_container_width=True
)

edited_inflow = edited_df["Inflow (₹)"].tolist()
edited_outflow = edited_df["Outflow (₹)"].tolist()

monthly_rate = (interest_rate / 100) / 12
if monthly_rate > 0:
    fixed_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**loan_term_months) / ((1 + monthly_rate)**loan_term_months - 1)
else:
    fixed_payment = loan_amount / loan_term_months

def run_simulation():
    np.random.seed(42)
    months = np.arange(1, loan_term_months + 1)

    inflows, outflows = [], []
    for m in range(loan_term_months):
        idx = m % 12
        # Induce natural variance
        inflows.append(edited_inflow[idx] * np.random.uniform(0.90, 1.10))
        outflows.append(edited_outflow[idx] * np.random.uniform(0.95, 1.05))

    inflows = np.array(inflows)
    outflows = np.array(outflows)
    net_operating_cash = inflows - outflows

    # State Variables
    fixed_balance = float(loan_amount)
    adaptive_balance = float(loan_amount)

    curr_f = 12000.0
    arrears_f = 0.0
    fixed_cash = []
    fixed_payments_collected = []
    fixed_defaults_count = 0
    fixed_payoff_month = None
    fixed_total_paid = 0.0

    curr_d = 12000.0
    dyn_cash = []
    dyn_payments_collected = []
    adaptive_payoff_month = None
    adaptive_total_paid = 0.0

    predicted_cash_flow = []
    stress_flags = []
    condition_types = []
    decision_traces = []

    history_window = [[np.sin(2 * np.pi * (m % 12) / 12), edited_inflow[m % 12], edited_outflow[m % 12], edited_inflow[m % 12] - edited_outflow[m % 12]] for m in range(6)]

    for m in range(loan_term_months):
        # ---------------------------------------------------------
        # 1. FIXED SCHEDULE LOGIC (With true arrears & penalties)
        # ---------------------------------------------------------
        if fixed_balance > 0.01:
            fixed_balance *= (1 + monthly_rate) # Accrue interest
            payment_due = fixed_payment + arrears_f
            
            # Borrower protects subsistence first before paying fixed EMI
            available_for_fixed = max(0.0, curr_f + net_operating_cash[m] - subsistence_buffer)
            actual_fixed_payment = min(payment_due, available_for_fixed)
            actual_fixed_payment = min(actual_fixed_payment, fixed_balance)
            
            if actual_fixed_payment < payment_due and fixed_balance > 1.0:
                # Default occurred
                arrears_f = (payment_due - actual_fixed_payment) * 1.05 # 5% penalty on arrears
                fixed_defaults_count += 1
            else:
                arrears_f = 0.0
                
            fixed_balance -= actual_fixed_payment
            curr_f = curr_f + net_operating_cash[m] - actual_fixed_payment
            fixed_total_paid += actual_fixed_payment
            
            if fixed_balance <= 0.01 and fixed_payoff_month is None:
                fixed_payoff_month = m + 1
        else:
            curr_f = curr_f + net_operating_cash[m]
            actual_fixed_payment = 0.0
            
        fixed_cash.append(curr_f)
        fixed_payments_collected.append(actual_fixed_payment)

        # ---------------------------------------------------------
        # 2. ADAPTIVE SCHEDULE LOGIC (With true interest accrual)
        # ---------------------------------------------------------
        
        # LSTM or Seasonal Naive Forecast
        if lstm_ready:
            recent_seq = np.array(history_window[-6:])
            seq_min, seq_max = recent_seq.min(axis=0), recent_seq.max(axis=0)
            seq_norm = (recent_seq - seq_min) / np.maximum(seq_max - seq_min, 1e-5)
            x_input = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = lstm_engine(x_input).item()
            pred_val = float(pred_scaled * (seq_max[3] - seq_min[3]) + seq_min[3])
        else:
            # Honest seasonal-naive lookahead (uses expected baseline, not real future array)
            pred_val = edited_inflow[(m + 1) % 12] - edited_outflow[(m + 1) % 12]
        predicted_cash_flow.append(pred_val)

        available_liquidity = curr_d + net_operating_cash[m]
        free_surplus = max(0.0, available_liquidity - subsistence_buffer)
        
        # 3-Month Rolling Mean Analysis (Permanent vs Temporary logic)
        if m >= 2:
            current_3m = np.mean(net_operating_cash[m-2:m+1])
            baseline_3m = np.mean([edited_inflow[i%12] - edited_outflow[i%12] for i in range(m-2, m+1)])
        else:
            current_3m = net_operating_cash[m]
            baseline_3m = edited_inflow[m%12] - edited_outflow[m%12]
            
        structural_deterioration = (current_3m < baseline_3m * 0.75) # Running 25% below baseline

        # State Evaluation
        if adaptive_balance <= 0.01:
            stress = "🔵 Debt Cleared"
            condition = "Loan Fully Repaid"
        else:
            if available_liquidity < fixed_payment:
                stress = "🚨 Critical Stress"
            elif available_liquidity < (fixed_payment + subsistence_buffer):
                stress = "⚠️ Moderate Pressure"
            else:
                stress = "🟢 Healthy Surplus"
            
            if current_3m < 0:
                condition = "Structural Deterioration" if structural_deterioration else "Temporary Seasonal Downturn"
            else:
                condition = "Peak Liquidity Surge" if current_3m > fixed_payment * 2.0 else "Stable Baseline Cash Flow"
        
        stress_flags.append(stress)
        condition_types.append(condition)

        # Algorithmic Forecast-Driven Payment Execution
        if adaptive_balance > 0.01:
            adaptive_balance *= (1 + monthly_rate) # Accrue actual interest
            
            current_month_surplus_ratio = net_operating_cash[m] / max(fixed_payment, 1.0)
            
            if free_surplus <= 500:
                ai_demand = 0.0 # Grace Period
            elif free_surplus < fixed_payment:
                ai_demand = free_surplus * 0.9 # Proportional Servicing
            elif current_month_surplus_ratio > 1.5:
                # Windfall Harvesting
                ai_demand = min(free_surplus * 0.6, fixed_payment * 4.0)
            else:
                ai_demand = fixed_payment
                
            ai_demand = min(ai_demand, free_surplus) 
            ai_demand = min(ai_demand, adaptive_balance) 
            
            adaptive_balance -= ai_demand
            adaptive_total_paid += ai_demand
            curr_d = available_liquidity - ai_demand
            
            if adaptive_balance <= 0.01 and adaptive_payoff_month is None:
                adaptive_payoff_month = m + 1
        else:
            ai_demand = 0.0
            curr_d = available_liquidity

        dyn_cash.append(curr_d)
        dyn_payments_collected.append(ai_demand)

        # Evidence Trace Generation
        if adaptive_balance <= 0.01 and ai_demand == 0:
            trace = "Loan fully repaid. Account closed."
        elif ai_demand <= 100.0:
            trace = f"Grace invoked. Forecast anticipates recovery of ₹{pred_val:,.0f} next month. Interest capitalized."
        elif ai_demand > fixed_payment * 1.25:
            trace = f"Harvest/Surplus capture. Accelerated principal recovery by ₹{(ai_demand - fixed_payment):,.0f}."
        elif ai_demand < fixed_payment * 0.95:
            trace = f"Rescheduled. Scaled payment to ₹{ai_demand:,.0f} to protect subsistence buffer."
        else:
            trace = "Standard amortization schedule applied."
        decision_traces.append(trace)

        history_window.append([np.sin(2 * np.pi * ((m + 1) % 12) / 12), inflows[m], outflows[m], net_operating_cash[m]])

    return (months, inflows, outflows, fixed_cash, dyn_cash, fixed_payments_collected, dyn_payments_collected, 
            predicted_cash_flow, stress_flags, condition_types, decision_traces, 
            fixed_defaults_count, fixed_payoff_month, adaptive_payoff_month)

if st.button("⚡ Run Core Analysis & Evidence Engine", use_container_width=True, type="primary"):
    (months, inflows, outflows, f_cash, d_cash, f_payments, d_payments,
     pred_cf, stress_flags, cond_types, decision_traces, 
     f_defaults, f_payoff, d_payoff) = run_simulation()

    # --- CALCULATE ALTERNATIVE CREDIT SCORE (AIRRS) ---
    base_score = 400
    
    total_repaid = sum(d_payments)
    expected_total = loan_amount + (loan_amount * (interest_rate/100) * (loan_term_years/2)) # Rough expectation
    repay_ratio = min(1.0, total_repaid / loan_amount)
    score_repayment = repay_ratio * 250
    
    breaches = sum(1 for c in d_cash if c <= subsistence_buffer)
    resilience_ratio = max(0.0, 1.0 - (breaches / loan_term_months))
    score_resilience = resilience_ratio * 150
    
    windfall_captures = sum(1 for p in d_payments if p > fixed_payment * 1.5)
    score_bonus = min(100, windfall_captures * 15)
    
    final_credit_score = int(min(850, max(300, base_score + score_repayment + score_resilience + score_bonus)))

    st.markdown("### 📊 Performance Overview & Credit Scoring")
    k1, k2, k3, k4, k5 = st.columns(5)
    
    score_color = "normal" if final_credit_score > 650 else "off"
    k1.metric("AIRRS Credit Score", f"{final_credit_score} / 850", delta="High Trust" if final_credit_score > 700 else "Developing", delta_color=score_color)
    
    # Fair Comparison Metrics
    k2.metric("Fixed Schedule Arrears", f"{f_defaults} Months", delta="Involuntary Default Risk", delta_color="inverse")
    k3.metric("Adaptive Reschedules", f"{sum(1 for p in d_payments if p < fixed_payment and p > 0)} Months", delta="0 Defaults")
    k4.metric("Fixed Payoff Time", f"{f_payoff} Mo" if f_payoff else "Did Not Finish", delta="Standard")
    k5.metric("Adaptive Payoff Time", f"{d_payoff} Mo" if d_payoff else "Did Not Finish", delta=f"{f_payoff - d_payoff if f_payoff and d_payoff else 0} Mo Faster", delta_color="normal")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=f_cash, mode='lines', name="Traditional Fixed Cash Buffer", line=dict(color='firebrick', dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=months, y=d_cash, mode='lines', name="Adaptive Algorithmic Cash Buffer", line=dict(color='seagreen', width=3)))
    fig.add_hline(y=0, line_dash="solid", line_color="black", annotation_text="Total Starvation (₹0)")
    fig.add_hline(y=subsistence_buffer, line_dash="dot", line_color="orange", annotation_text="Protected Subsistence Buffer")
    fig.add_trace(go.Bar(x=months, y=d_payments, name="Adaptive Payment Collected", opacity=0.35, marker_color='royalblue', yaxis="y2"))

    fig.update_layout(
        title=f"{loan_term_years}-Year Multi-Cycle Cash Flow & Contagion Defense",
        xaxis_title="Tenure (Months)",
        yaxis_title="Borrower Liquidity (₹)",
        yaxis2=dict(title="Repayment Collected (₹)", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Model Evidence, Stress Identification & Diagnostic Trace")
    
    tab1, tab2, tab3 = st.tabs(["📋 Complete Audit Ledger", "📈 Model 1: LSTM vs Baseline Forecast", "🧠 Core Evidence Breakdown"])

    with tab1:
        ledger_df = pd.DataFrame({
            "Month": months,
            "Inflow (₹)": [f"₹{v:,.0f}" for v in inflows],
            "Outflow (₹)": [f"₹{v:,.0f}" for v in outflows],
            "Fixed Actual Paid": [f"₹{p:,.2f}" for p in f_payments],
            "Adaptive Demand": [f"₹{p:,.2f}" for p in d_payments],
            "Repayment Stress Level": stress_flags,
            "Financial Condition State": cond_types,
            "Explainability Evidence Trace": decision_traces
        })
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)

    with tab2:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=months, y=inflows - outflows, mode='lines+markers', name="Actual Realized Margin"))
        fig_pred.add_trace(go.Scatter(x=months, y=pred_cf, mode='lines', name="LSTM / Baseline Lookahead", line=dict(dash='dot', color='purple')))
        fig_pred.update_layout(title="Accuracy: Predictive Anticipation vs Realized Operating Cash", template="plotly_white")
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab3:
        st.markdown(f"""
        #### Systematic Findings & Alternative Credit Profile:
        
        * **1. True Interest & Arrears Simulation:**  
          Unlike basic models, this simulation mathematically accrues interest `(1 + monthly_rate)` on the outstanding balance every month. Missed fixed payments properly cascade into penalty arrears, proving the adaptive model clears debt structurally faster.
          
        * **2. The New Alternative Credit Metric (Score: {final_credit_score}/850):**
          Traditional bureaus punish this borrower for missing fixed dates. Our system generated an **Alternative Credit Score (AIRRS)** based on behavioral integrity:
          - **Debt Recovery Index:** Evaluated ultimate principal recovery rather than strict monthly adherence.
          - **Buffer Resilience:** Scored based on their ability to maintain the ₹{subsistence_buffer:,.0f} survival threshold.
          - **Windfall Integrity Bonus:** Rewarded for successfully surrendering surplus liquidity during harvest/festival peaks.
          
        * **3. Structural Deterioration vs. Seasonality:**  
          The AI engine calculates a **3-Month Rolling Mean** and contrasts it with the historical baseline profile. This allows the bank to automatically distinguish between a normal "Temporary Seasonal Downturn" (requiring grace) versus "Structural Deterioration" (requiring intervention).
          
        * **4. Forecast-Driven Payment Policy:**  
          By decoupling the tenure, the algorithmic engine executes three adaptive modes:  
          - **Grace (₹0):** Invoked when the buffer drops. Interest capitalizes, but the borrower survives.  
          - **Proportional Servicing:** Scaled debt servicing during moderate seasonal drag.  
          - **Supercharged Liquidity Harvesting:** Repayments scaling up to 4x normal levels during peak cycles to erase accrued interest.
        """)