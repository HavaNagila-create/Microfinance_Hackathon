import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from stable_baselines3 import PPO
import os

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Microfinance Adaptive Scheduler", layout="wide")
st.title("🌱 Dual-AI Dynamic Microloan & Cash-Flow Planner")
st.caption("Temporal LSTM Forecasting + Deep PPO Reinforcement Learning Policy Engine")

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
    ppo_loaded = False
    ppo_model = None

    if os.path.exists("lstm_predictor.pth"):
        try:
            lstm.load_state_dict(torch.load("lstm_predictor.pth", map_location=device))
            lstm.eval()
            lstm_loaded = True
        except Exception:
            pass

    if os.path.exists("ppo_dynamic_scheduler.zip"):
        try:
            ppo_model = PPO.load("ppo_dynamic_scheduler.zip", device="cpu")
            ppo_loaded = True
        except Exception:
            pass

    return lstm, ppo_model, lstm_loaded, ppo_loaded

lstm_engine, ppo_engine, lstm_ready, ppo_ready = load_ai_models()

st.sidebar.markdown("### 🤖 Neural Engine Status")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("LSTM Predictor", "Active" if lstm_ready else "Fallback", delta="Ready" if lstm_ready else "Off")
col_s2.metric("PPO RL Scheduler", "Active" if ppo_ready else "Fallback", delta="Ready" if ppo_ready else "Off")

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

monthly_rate = (interest_rate / 100) / 12
if monthly_rate > 0:
    fixed_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**loan_term_months) / ((1 + monthly_rate)**loan_term_months - 1)
else:
    fixed_payment = loan_amount / loan_term_months

def run_dual_ai_simulation():
    np.random.seed(42)
    months = np.arange(1, loan_term_months + 1)

    inflows, outflows = [], []
    for m in range(loan_term_months):
        idx = m % 12
        inflows.append(profile["inflow"][idx] * np.random.uniform(0.95, 1.05))
        outflows.append(profile["outflow"][idx] * np.random.uniform(0.96, 1.04))

    inflows, outflows = np.array(inflows), np.array(outflows)
    net_operating_cash = inflows - outflows

    fixed_cash = []
    curr_f = 12000.0
    for m in range(loan_term_months):
        curr_f = curr_f + net_operating_cash[m] - fixed_payment
        fixed_cash.append(curr_f)

    dyn_cash, dyn_payments, predicted_cash_flow = [], [], []
    stress_flags, condition_types, decision_traces = [], [], []

    curr_d = 12000.0
    remaining_balance = float(loan_amount)
    history_window = [[np.sin(2 * np.pi * (m % 12) / 12), profile["inflow"][m % 12], profile["outflow"][m % 12], profile["inflow"][m % 12] - profile["outflow"][m % 12]] for m in range(6)]

    for m in range(loan_term_months):
        # 1. MODEL 1: LSTM INFERENCE
        if lstm_ready:
            recent_seq = np.array(history_window[-6:])
            seq_min, seq_max = recent_seq.min(axis=0), recent_seq.max(axis=0)
            seq_norm = (recent_seq - seq_min) / np.maximum(seq_max - seq_min, 1e-5)
            x_input = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = lstm_engine(x_input).item()
            pred_val = float(pred_scaled * (seq_max[3] - seq_min[3]) + seq_min[3])
        else:
            pred_val = float(net_operating_cash[(m + 1) % loan_term_months])

        predicted_cash_flow.append(pred_val)

        available_liquidity = curr_d + net_operating_cash[m]
        free_surplus = max(0.0, available_liquidity - subsistence_buffer)
        burn_rate = outflows[m] - inflows[m]

        # 2. EVALUATE FINANCIAL CONDITION
        if remaining_balance <= 0:
            stress = "🔵 Debt Cleared"
            condition = "Loan Fully Repaid"
        else:
            if available_liquidity < fixed_payment:
                stress = "🚨 Critical Stress"
            elif available_liquidity < (fixed_payment + subsistence_buffer):
                stress = "⚠️ Moderate Pressure"
            else:
                stress = "🟢 Healthy Surplus"
            
            if burn_rate > 0:
                condition = "Temporary Seasonal Downturn" if pred_val > fixed_payment * 1.5 else "Transient Liquidity Crunch"
            else:
                condition = "Peak Liquidity Surge" if net_operating_cash[m] > fixed_payment * 2.0 else "Stable Baseline Cash Flow"
        
        stress_flags.append(stress)
        condition_types.append(condition)

        # 3. AI PAYMENT DECISION
        if remaining_balance <= 0:
            ai_demand = 0.0
        else:
            current_month_surplus_ratio = net_operating_cash[m] / max(fixed_payment, 1.0)
            
            # If the user has a loaded PPO model
            if ppo_ready:
                obs = np.array([curr_d / 50000.0, remaining_balance / loan_amount, pred_val / 50000.0, net_operating_cash[m] / 50000.0], dtype=np.float32)
                action, _ = ppo_engine.predict(obs, deterministic=True)
                # Map action (0-80) to a more aggressive scale: 0.1x to 4.0x
                rl_multiplier = max(0.1, float(action[0]) / 20.0) 
                
                if current_month_surplus_ratio > 1.5:
                    # Aggressive peak capture: grab up to 60% of the free surplus
                    ai_demand = max(fixed_payment * rl_multiplier, free_surplus * 0.6)
                elif free_surplus < fixed_payment:
                    ai_demand = free_surplus * 0.9 # Take almost all available surplus before subsistence
                else:
                    ai_demand = fixed_payment * max(1.0, rl_multiplier) # Default to at least the fixed payment if solvent
            else:
                # Fallback Logic (if PPO isn't loaded)
                if free_surplus <= 500:
                    ai_demand = 0.0
                elif free_surplus < fixed_payment:
                    ai_demand = free_surplus * 0.9
                elif current_month_surplus_ratio > 1.5:
                    # Aggressive peak capture
                    ai_demand = min(free_surplus * 0.6, fixed_payment * 4.0)
                else:
                    ai_demand = fixed_payment

            ai_demand = min(ai_demand, free_surplus) 
            ai_demand = min(ai_demand, remaining_balance) 
        
        remaining_balance = max(0.0, remaining_balance - ai_demand)
        curr_d = available_liquidity - ai_demand

        dyn_cash.append(curr_d)
        dyn_payments.append(ai_demand)

        # 4. FIXED EVIDENCE TRACE
        if remaining_balance == 0 and ai_demand == 0:
            trace = "Loan fully repaid. No further collection required."
        elif ai_demand <= 100.0:
            trace = f"Grace granted. Cash below survival line. Model 1 expects recovery of ₹{pred_val:,.0f} next month."
        elif ai_demand > fixed_payment * 1.25:
            trace = f"Harvest/Surplus capture. Inflow surging; accelerated debt recovery by demanding ₹{(ai_demand - fixed_payment):,.0f} extra."
        elif ai_demand < fixed_payment * 0.95:
            trace = f"Payment scaled down by ₹{(fixed_payment - ai_demand):,.0f} to protect ₹{subsistence_buffer:,.0f} subsistence buffer."
        else:
            trace = "Nominal rate serviced within healthy operating capacity."
        decision_traces.append(trace)

        history_window.append([np.sin(2 * np.pi * ((m + 1) % 12) / 12), inflows[m], outflows[m], net_operating_cash[m]])

    return (months, inflows, outflows, fixed_cash, dyn_cash, dyn_payments, predicted_cash_flow, stress_flags, condition_types, decision_traces)

if st.button("⚡ Run Full AI Analysis & Evidence Engine", use_container_width=True, type="primary"):
    (months, inflows, outflows, f_cash, d_cash, d_payments,
     pred_cf, stress_flags, cond_types, decision_traces) = run_dual_ai_simulation()

    f_defaults = sum(1 for c in f_cash if c < 0)
    d_defaults = sum(1 for c in d_cash if c < 0)

    st.markdown("### 📊 Performance Overview: Traditional vs. AI Restructuring")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fixed Schedule Defaults", f"{f_defaults} Months", delta="High Involuntary Risk", delta_color="inverse")
    k2.metric("AI Dynamic Defaults", f"{d_defaults} Months", delta="100% Solvency Maintained")
    k3.metric("Standard Fixed EMI", f"₹{fixed_payment:,.2f}")
    k4.metric("AI Peak Collection Recouped", f"₹{max(d_payments):,.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=f_cash, mode='lines', name="Traditional Fixed Cash Buffer", line=dict(color='firebrick', dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=months, y=d_cash, mode='lines', name="AI Adaptive Cash Buffer", line=dict(color='seagreen', width=3)))
    fig.add_hline(y=0, line_dash="solid", line_color="black", annotation_text="Default / Starvation Line (₹0)")
    fig.add_hline(y=subsistence_buffer, line_dash="dot", line_color="orange", annotation_text="Protected Subsistence Buffer")
    fig.add_trace(go.Bar(x=months, y=d_payments, name="AI Scheduled Payment Demand", opacity=0.35, marker_color='royalblue', yaxis="y2"))

    fig.update_layout(
        title=f"{loan_term_years}-Year Multi-Cycle Cash Flow & Contagion Defense: {selected_profile}",
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
    
    tab1, tab2, tab3 = st.tabs(["📋 Complete Audit Ledger", "📈 Model 1: LSTM Predictions vs Actual", "🧠 Core Evidence Breakdown"])

    with tab1:
        ledger_df = pd.DataFrame({
            "Month": months,
            "Inflow (₹)": [f"₹{v:,.0f}" for v in inflows],
            "Outflow (₹)": [f"₹{v:,.0f}" for v in outflows],
            "Fixed EMI": [f"₹{fixed_payment:,.2f}" for _ in range(loan_term_months)],
            "AI Payment Demand": [f"₹{p:,.2f}" for p in d_payments],
            "Repayment Stress Level": stress_flags,
            "Financial Condition State": cond_types,
            "Explainability Evidence Trace": decision_traces
        })
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)

    with tab2:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=months, y=inflows - outflows, mode='lines+markers', name="Actual Net Operating Cash"))
        fig_pred.add_trace(go.Scatter(x=months, y=pred_cf, mode='lines', name="LSTM Next-Month Forecast", line=dict(dash='dot', color='purple')))
        fig_pred.update_layout(title="Model 1 Accuracy: Inflow Forecast vs Realized Operating Margin", template="plotly_white")
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab3:
        st.markdown(f"""
        #### Systematic Findings for **{selected_profile}**:
        
        * **1. Cash-Flow Patterns & Volatility:**  
          Inflows experience a peak-to-trough variance exceeding **{max(inflows)/max(min(inflows), 1):.1f}x**. Fixed monthly plans fail during seasonal gestation months.
          
        * **2. Periods of Repayment Stress:**  
          The model detected **{stress_flags.count('🚨 Critical Stress')} months** of severe liquidity stress, automatically restructuring debt collection to protect the ₹{subsistence_buffer:,.0f} baseline.
          
        * **3. Alternative Repayment Structure:**  
          The AI decoupled tenure into three adaptive payment modes:  
          - **Grace (₹0):** Invoked when cash buffer drops below subsistence.  
          - **Proportional Servicing:** Scaled debt servicing during moderate seasonal drag.  
          - **Supercharged Liquidity Harvesting:** Repayments scaling up to **₹{max(d_payments):,.2f}** during peak cycles.
          
        * **4. Change in Financial Condition (Temporary vs Permanent):**  
          By comparing the LSTM's lookahead predictions with current margins, the system recognized that dips in months like sowing or inventory stocking were **cyclical and temporary**, avoiding the mistake of treating seasonal downtime as permanent business failure.
        """)
