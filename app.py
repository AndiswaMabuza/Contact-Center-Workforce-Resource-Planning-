import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from datetime import timedelta, datetime
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Workforce & Resource Planning App", layout="wide")

@st.cache_data
def simulate_data():
    time_index = pd.date_range("2024-01-01", "2025-07-31 23:45:00", freq="15min")
    channels = ["Voice", "Chat", "Email"]
    df_calls = pd.DataFrame([
        [ts, ch, np.random.poisson((20 if ch=="Voice" else 10) * (1.5 if 9<=ts.hour<=18 else 0.5))]
        for ts in time_index for ch in channels
    ], columns=["Timestamp", "Channel", "Volume"])

    agent_ids = [f"Agent_{i}" for i in range(1, 101)]
    dates = pd.date_range("2024-01-01", "2025-07-31", freq="D")
    shifts, perf = [], []
    for date in dates:
        for agent in agent_ids:
            shift_start = datetime.combine(date, datetime.min.time()) + timedelta(hours=random.choice([8, 9, 10]))
            shifts.append([agent, date, shift_start, shift_start + timedelta(hours=8),
                           random.choice(["EN","ES","FR"]), random.choice(channels)])
            perf.append([
                agent, date,
                np.random.normal(0.9, 0.05),  # Adherence
                np.random.normal(0.75, 0.1),  # Occupancy
                np.random.normal(0.2, 0.05),  # Shrinkage
                np.random.normal(300, 30)     # AHT
            ])
    df_shifts = pd.DataFrame(shifts, columns=["Agent_ID","Date","Shift_Start","Shift_End","Language","Channel"])
    df_perf = pd.DataFrame(perf, columns=["Agent_ID","Date","Adherence","Occupancy","Shrinkage","AHT"])
    return df_calls, df_shifts, df_perf

df_calls, df_shifts, df_perf = simulate_data()

# ---------------- Sidebar Controls -------------------
st.sidebar.title("🔧 Forecast & Demand Simulator")
channel_select = st.sidebar.selectbox("Select Channel", ["Voice", "Chat", "Email"])
forecast_days = st.sidebar.slider("Forecast Period (days)", 7, 60, 14)
expected_contacts = st.sidebar.number_input("Expected Total Contacts", 500, 10000, 2000, step=500)
spread_days = st.sidebar.slider("Over How Many Days?", 1, 30, 7)

# SLA / Business Targets
sla_target, aht_target = 0.80, 300
occupancy_target, shrinkage_target = 0.75, 0.20
aht_lookup = {"Voice": 300, "Chat": 180, "Email": 240}

# ---------------- App Header ----------------------------
st.title("Contact Center Workforce & Resource Planning")
st.markdown("Interactive simulation for forecasting, scheduling, monitoring, and performance recommendations.")

# ---------------- Historical EDA -------------------------
st.header("Historical Demand & Performance")
col1, col2 = st.columns(2)

with col1:
    df_calls["Hour"] = df_calls["Timestamp"].dt.hour
    avg_vol = df_calls.groupby(["Hour","Channel"])["Volume"].mean().reset_index()
    fig = px.line(avg_vol, x="Hour", y="Volume", color="Channel", title="Avg Call Volume by Hour")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    daily_shrink = df_perf.groupby("Date")["Shrinkage"].mean().reset_index()
    fig = px.line(daily_shrink, x="Date", y="Shrinkage", title="Daily Shrinkage")
    fig.add_hline(y=shrinkage_target, line_dash="dash", annotation_text="Target")
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    daily_occ = df_perf.groupby("Date")["Occupancy"].mean().reset_index()
    fig = px.line(daily_occ, x="Date", y="Occupancy", title="Occupancy Trend")
    fig.add_hline(y=occupancy_target, line_dash="dash", annotation_text="Target")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    daily_adh = df_perf.groupby("Date")["Adherence"].mean().reset_index()
    fig = px.line(daily_adh, x="Date", y="Adherence", title="Adherence Trend")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Forecasting ---------------------------
st.header("Demand Forecasting")
df_channel = df_calls[df_calls["Channel"]==channel_select]
daily = df_channel.groupby(df_channel["Timestamp"].dt.date)["Volume"].sum().reset_index()
daily.columns = ["ds","y"]

m = Prophet()
m.fit(daily)
future = m.make_future_dataframe(periods=forecast_days)
fc = m.predict(future)

viz = pd.concat([
    daily.assign(type="Actual"),
    fc[["ds","yhat"]].rename(columns={"yhat":"y"}).assign(type="Forecast")
])

fig_fc = px.line(viz, x="ds", y="y", color="type",
                 title=f"{channel_select} Forecast for next {forecast_days} days")
fig_fc.add_vline(x=pd.to_datetime(daily["ds"].max()), line_dash="dash", annotation_text="Forecast Starts")
fig_fc.update_layout(xaxis_title="Date", yaxis_title="Volume")
st.plotly_chart(fig_fc, use_container_width=True)

# Agent requirement based on anticipated demand
aht_secs = aht_lookup.get(channel_select, 300)
calls_per_agent_day = int((8*3600)/aht_secs)
agents_needed = int(np.ceil((expected_contacts / (calls_per_agent_day * spread_days)) / (1 - shrinkage_target)))
st.subheader("Anticipated Demand Simulation")
st.markdown(f"- **Expected contacts**: {expected_contacts} over {spread_days} days")
st.markdown(f"- **Calls per agent/day**: {calls_per_agent_day}")
st.markdown(f"- **Agents required (shrinkage adjusted)**: **{agents_needed}**")

# ---------------- Scheduling ----------------------------
st.header("Scheduling Simulation")
forecast_day = fc["ds"].iloc[-1]
req_calls = int(fc[fc["ds"]==forecast_day]["yhat"].values[0])
req_agents_fc = int(np.ceil((req_calls / calls_per_agent_day) / (1 - shrinkage_target)))
available = df_shifts[
    (df_shifts["Date"]==forecast_day)&
    (df_shifts["Channel"]==channel_select)&
    (df_shifts["Language"]=="EN")
]

st.metric("Forecast Date", forecast_day)
st.metric("Forecasted Calls", req_calls)
st.metric("Agents Required", req_agents_fc)
st.metric("Available EN Agents", len(available))

# ---------------- Real-time Monitoring --------------------
st.header("Real-Time Monitoring Simulation")
realtime = pd.DataFrame({
    "Hour": list(range(9,18)),
    "AHT":[round(x,1) for x in np.random.normal(310,15,9)],
    "Occupancy":[round(x,2) for x in np.random.normal(0.78,0.03,9)],
    "SLA":[round(x,2) for x in np.random.normal(0.82,0.05,9)]
})
st.dataframe(realtime)

# ---------------- Agent-Level Performance ------------------
st.header("Agent-Level Performance Overview")
agent_perf = df_perf.groupby("Agent_ID")[["Adherence","Occupancy","Shrinkage","AHT"]].mean().reset_index()
fig_ag = px.scatter(agent_perf, x="Occupancy", y="Adherence",
                    size="Shrinkage", color="AHT",
                    hover_name="Agent_ID", title="Agent Performance")
st.plotly_chart(fig_ag, use_container_width=True)

# ---------------- Issues & Actions Table -------------------
st.header("Issues & Actions")
summary = df_perf.groupby("Date")[["Adherence","Occupancy","Shrinkage","AHT"]].mean().mean()
issues = []

# Shrinkage spikes
high_shrink = df_perf[df_perf["Shrinkage"]>0.25]
if summary["Shrinkage"] > shrinkage_target:
    top = high_shrink.groupby("Agent_ID").size().nlargest(3)
    issues.append({
        "Issue": f"Shrinkage >25% occurred {top.values[0]} times (most frequent: {top.index[0]})",
        "Action": f"Flag {top.index[0]} for attendance review; shift training out of peak hours"
    })

# Low adherence
low_adh = df_perf[df_perf["Adherence"]<0.8]
if summary["Adherence"]<0.85:
    top = low_adh.groupby("Agent_ID").size().nlargest(3)
    issues.append({
        "Issue": f"Adherence <80% occurred {top.values[0]} times (worst: {top.index[0]})",
        "Action": f"Assign agent {top.index[0]} mandatory coaching this week; send daily adherence alerts"
    })

# Low occupancy
low_occ = df_perf[df_perf["Occupancy"]<0.6]
if summary["Occupancy"]<occupancy_target:
    top = low_occ.groupby("Agent_ID").size().nlargest(3)
    issues.append({
        "Issue": f"Occupancy <60% occurred {top.values[0]} times (underutil: {top.index[0]})",
        "Action": f"Reassign agent {top.index[0]} to multi-channel tasks or reduce shifts by 1 day"
    })

# High AHT
high_aht = df_perf[df_perf["AHT"]>360]  # 6 min
if summary["AHT"]> aht_target:
    top = high_aht.groupby("Agent_ID").size().nlargest(3)
    issues.append({
        "Issue": f"AHT above 6 min occurred {top.values[0]} times (longest: {top.index[0]})",
        "Action": f"Review calls of agent {top.index[0]}; conduct efficiency refresher training"
    })

issues_df = pd.DataFrame(issues)
edited = st.data_editor(issues_df, num_rows="fixed", use_container_width=True)
st.markdown("Managers can review issues and edit actions as needed.")

# ---------------- Summary Section ------------------------
st.header("Performance Summary (Jan 2024 – Jul 2025)")
st.markdown(f"- **Adherence**: {summary['Adherence']:.2%}")
st.markdown(f"- **Occupancy**: {summary['Occupancy']:.2%}")
st.markdown(f"- **Shrinkage**: {summary['Shrinkage']:.2%}")
st.markdown(f"- **AHT**: {summary['AHT']:.0f} seconds")

