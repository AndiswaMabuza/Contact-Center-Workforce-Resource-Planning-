import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from datetime import timedelta, datetime
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Contact Center Workforce & Resource Planning", layout="wide")

# --------------- Simulate or Load Data --------------------
@st.cache_data
def simulate_data():
    time_index = pd.date_range(start="2024-01-01", end="2025-07-31 23:45:00", freq="15min")
    channels = ["Voice", "Chat", "Email"]
    data = []
    for ts in time_index:
        for ch in channels:
            base = 20 if ch == "Voice" else 10
            hour = ts.hour
            modifier = 1.5 if 9 <= hour <= 18 else 0.5
            volume = np.random.poisson(base * modifier)
            data.append([ts, ch, volume])
    df_calls = pd.DataFrame(data, columns=["Timestamp", "Channel", "Volume"])

    agent_ids = [f"Agent_{i}" for i in range(1, 101)]
    dates = pd.date_range("2024-01-01", "2025-07-31", freq="D")

    shifts = []
    perf = []
    for date in dates:
        for agent in agent_ids:
            shift_start = datetime.combine(date, datetime.min.time()) + timedelta(hours=random.choice([8, 9, 10]))
            shift_end = shift_start + timedelta(hours=8)
            language = random.choice(["EN", "ES", "FR"])
            channel = random.choice(channels)
            shifts.append([agent, date, shift_start, shift_end, language, channel])

            adherence = np.random.normal(0.9, 0.05)
            occupancy = np.random.normal(0.75, 0.1)
            shrinkage = np.random.normal(0.2, 0.05)
            aht = np.random.normal(300, 30)
            perf.append([agent, date, adherence, occupancy, shrinkage, aht])

    df_shifts = pd.DataFrame(shifts, columns=["Agent_ID", "Date", "Shift_Start", "Shift_End", "Language", "Channel"])
    df_perf = pd.DataFrame(perf, columns=["Agent_ID", "Date", "Adherence", "Occupancy", "Shrinkage", "AHT"])

    return df_calls, df_shifts, df_perf

df_calls, df_shifts, df_perf = simulate_data()

# ---------------- Sidebar Controls -------------------
st.sidebar.title("🔧 Forecast Simulator")
channel_select = st.sidebar.selectbox("Select Channel", ["Voice", "Chat", "Email"])
forecast_period = st.sidebar.slider("Forecast Days", 7, 60, 14)

sla_target = 0.8
aht_target = 300
occupancy_target = 0.75
shrinkage_target = 0.2

# ----------------- Page Title ------------------------
st.title("📞 Contact Center Workforce & Resource Planning")
st.markdown("This interactive dashboard simulates forecasting, scheduling, and performance tracking for a multi-channel contact center.")

# ----------------- EDA Section -----------------------
st.header("📊 Historical Demand and Performance")

col1, col2 = st.columns(2)

with col1:
    df_calls["Hour"] = df_calls["Timestamp"].dt.hour
    avg_volume = df_calls.groupby(["Hour", "Channel"])["Volume"].mean().reset_index()
    fig = px.line(avg_volume, x="Hour", y="Volume", color="Channel", title="Average Volume by Hour")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    daily_shrink = df_perf.groupby("Date")["Shrinkage"].mean().reset_index()
    fig = px.line(daily_shrink, x="Date", y="Shrinkage", title="Daily Shrinkage Trend")
    fig.add_hline(y=shrinkage_target, line_dash="dash", annotation_text="Target")
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    daily_occupancy = df_perf.groupby("Date")["Occupancy"].mean().reset_index()
    fig = px.line(daily_occupancy, x="Date", y="Occupancy", title="Occupancy Trend")
    fig.add_hline(y=occupancy_target, line_dash="dash", annotation_text="Target")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    daily_adherence = df_perf.groupby("Date")["Adherence"].mean().reset_index()
    fig = px.line(daily_adherence, x="Date", y="Adherence", title="Adherence Trend")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Forecasting Section ----------------
st.header("🔮 Demand Forecasting")

channel_data = df_calls[df_calls["Channel"] == channel_select].copy()
daily_calls = channel_data.groupby(channel_data["Timestamp"].dt.date)["Volume"].sum().reset_index()
daily_calls.columns = ["ds", "y"]

model = Prophet()
model.fit(daily_calls)
future = model.make_future_dataframe(periods=forecast_period)
forecast = model.predict(future)

fig1 = model.plot(forecast)
st.pyplot(fig1)

# ---------------- Scheduling Section -----------------
st.header("🧭 Scheduling Simulation")

forecast_day = future["ds"].iloc[-1]
required_calls = int(forecast[forecast["ds"] == forecast_day]["yhat"].values[0])
required_agents = int(np.ceil((required_calls / 96) / (1 - shrinkage_target)))

available_agents = df_shifts[
    (df_shifts["Date"] == forecast_day) &
    (df_shifts["Channel"] == channel_select) &
    (df_shifts["Language"] == "EN")  # Fixed to English for simplicity
]

st.metric("Forecasted Calls", required_calls)
st.metric("Required Agents (adjusted)", required_agents)
st.metric("Available Scheduled Agents (EN)", len(available_agents))

# ---------------- Real-time Monitoring ---------------
st.header("🕒 Real-Time Monitoring Simulation")

hours = list(range(9, 18))
realtime = pd.DataFrame({
    "Hour": hours,
    "AHT": np.random.normal(310, 15, len(hours)),
    "Occupancy": np.random.normal(0.78, 0.03, len(hours)),
    "SLA": np.random.normal(0.82, 0.05, len(hours))
})
st.dataframe(realtime)

# ---------------- Agent-Level Performance ------------
st.header("👥 Agent-Level Performance")

agent_perf = df_perf.groupby("Agent_ID")[["Adherence", "Occupancy", "Shrinkage", "AHT"]].mean().reset_index()

fig = px.scatter(agent_perf, x="Occupancy", y="Adherence",
                 size="Shrinkage", color="AHT", hover_name="Agent_ID",
                 title="Agent-Level Performance Overview")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Recommendations & Summary ----------
st.header("🧠 Summary & Recommendations")

summary = df_perf.groupby("Date")[["Adherence", "Occupancy", "Shrinkage", "AHT"]].mean().mean().to_dict()
st.subheader("📋 Summary (Jan 2024 - Jul 2025)")
st.markdown(f"""
- **Avg Adherence**: {summary['Adherence']:.2%}  
- **Avg Occupancy**: {summary['Occupancy']:.2%}  
- **Avg Shrinkage**: {summary['Shrinkage']:.2%}  
- **Avg AHT**: {summary['AHT']:.0f} seconds
""")

# Detailed recommendations
high_shrink_agents = df_perf[df_perf["Shrinkage"] > 0.25]["Agent_ID"].value_counts().head(3).index.tolist()
low_adherence_agents = df_perf[df_perf["Adherence"] < 0.8]["Agent_ID"].value_counts().head(3).index.tolist()
underutilized_agents = df_perf[df_perf["Occupancy"] < 0.6]["Agent_ID"].value_counts().head(3).index.tolist()

st.subheader("📌 Recommendations")

if summary["Shrinkage"] > shrinkage_target:
    st.markdown(f"- 🚨 **Shrinkage too high** — frequent offenders: `{', '.join(high_shrink_agents)}`. Review their absentee logs and shift compliance.")

if summary["Adherence"] < 0.85:
    st.markdown(f"- ⏰ **Adherence issues** — `{', '.join(low_adherence_agents)}` consistently below 80%. Recommend targeted coaching or adherence alerting.")

if summary["Occupancy"] < occupancy_target:
    st.markdown(f"- 💤 **Underutilization** — `{', '.join(underutilized_agents)}` may be overscheduled or under-assigned. Suggest shift or task redistribution.")

if summary["AHT"] > aht_target:
    st.markdown(f"- 🕒 **AHT above target**. Avg: {summary['AHT']:.0f}s. Recommend QA review and refresher training on call efficiency.")

