import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# COLORS
PRIMARY = "#176B87"
ACCENT = "#64CCC5"
DANGER = "#C00000"
sns.set_theme(style="whitegrid")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("full_pipeline_balanced_output.csv")

df = load_data()

# =====================================================
# EMBEDDED MAPPINGS
# =====================================================
mapping_data = {
    "global_category_map": {
        "0": "AUDIO",
        "1": "CAMERA",
        "2": "HOME AUTOMATION"
    },
    "device_map": {
        "0": "simcam", "1": "homeeyecam", "2": "arloqcam",
        "3": "arlobasestationcam", "4": "luohecam", "5": "amcrest",
        "6": "heimvisioncam", "7": "dlinkcam", "8": "eufyhomebase",
        "9": "netatmocam", "10": "nestcam", "11": "sonosone",
        "12": "nestmini", "13": "echospot", "14": "echostudio",
        "15": "echodot", "16": "boruncam", "17": "roomba",
        "18": "philipshue", "19": "globelamp", "20": "amazonplug",
        "21": "heimvisionlamp", "22": "atomicoffeemaker",
        "23": "teckin2", "24": "yutron1", "25": "yutron2",
        "26": "teckin1", "27": "smartboard", "28": "heimvision"
    }
}

global_map = {int(k): v for k, v in mapping_data["global_category_map"].items()}
device_map = {int(k): v for k, v in mapping_data["device_map"].items()}

df["global_label"] = df["global_category"].map(global_map)
df["device_label"] = df["device"].map(device_map)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🔧 Dashboard Controls")
st.title("📡 Smart Home QoS & Trust Monitoring Dashboard")

global_sel = st.sidebar.selectbox(
    "Select Global Category:",
    sorted(df["global_label"].unique())
)
df_filtered = df[df["global_label"] == global_sel]

device_opts = df_filtered[["device", "device_label"]].drop_duplicates().sort_values("device_label")
device_sel_label = st.sidebar.selectbox("Select Device:", device_opts["device_label"])
device_sel = device_opts[device_opts["device_label"] == device_sel_label]["device"].values[0]

num_packets = st.sidebar.selectbox("Packets to analyze:", [100, 300, 500, 1000, 2000])
smooth_window = st.sidebar.slider("Smoothing Window:", 5, 100, 20)

metric_map = {
    "Delay": "qos_delay",
    "Jitter": "qos_jitter",
    "Throughput": "qos_throughput",
    "Bandwidth": "qos_bandwidth",
    "Packet Size": "qos_packet_size",
    "Frame Size": "qos_frame_size",
    "Trust Score": "trust_score",
    "Attack Probability": "attack_prob"
}

metric_label = st.sidebar.selectbox("Select Metric:", list(metric_map.keys()))
metric_col = metric_map[metric_label]

# =====================================================
# FILTER DEVICE DATA
# =====================================================
df_dev = df_filtered[df_filtered["device"] == device_sel].copy()
df_dev = df_dev.sort_values("epoch_timestamp").reset_index(drop=True)
df_dev = df_dev.head(num_packets)
df_dev[f"{metric_col}_smooth"] = df_dev[metric_col].rolling(smooth_window).mean()

# =====================================================
# MAIN PLOT
# =====================================================
st.subheader(f"📈 {metric_label} (Smoothed) — {device_sel_label}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_dev.index, df_dev[f"{metric_col}_smooth"], linewidth=2.4, color=PRIMARY)
ax.set_title(metric_label, fontsize=20)
ax.set_xlabel("Packet Index")
ax.grid(alpha=0.25)
st.pyplot(fig)

st.markdown("---")

# =====================================================
# TABS
# =====================================================
tabs = st.tabs([
    "Trust Over Time",                    # 0
    "Attack Probability Over Time",       # 1
    "Combined Trust + Attack",            # 2
    "QoS Heatmap",                        # 3
    "Raw QoS Table",                      # 4
    "Trust vs Selected QoS",              # 5
    "Threat Gauge",                       # 6
    "Device Risk Ranking",                # 7
    "Smart Home Attack Overview",         # 8 (new)
    "Risk Radar Chart",                   # 9
    "Attack Type Prediction",             # 10
    "Attack Type Signatures (QoS Table)", # 11
    "Most Likely Attack Type (Ranking)"   # 12
])

# =====================================================
# TAB 0 — TRUST OVER TIME
# =====================================================
with tabs[0]:
    st.subheader("📉 Trust Score (Smoothed)")
    df_dev["trust_smooth"] = df_dev["trust_score"].rolling(smooth_window).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_dev.index, df_dev["trust_smooth"], color=ACCENT, linewidth=2)
    ax.set_xlabel("Packet Index")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

# =====================================================
# TAB 1 — ATTACK PROBABILITY OVER TIME
# =====================================================
with tabs[1]:
    st.subheader("📉 Attack Probability (Smoothed)")
    df_dev["attack_smooth"] = df_dev["attack_prob"].rolling(smooth_window).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_dev.index, df_dev["attack_smooth"], color=DANGER, linewidth=2)
    ax.set_xlabel("Packet Index")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

# =====================================================
# TAB 2 — COMBINED TRUST + ATTACK
# =====================================================
with tabs[2]:
    st.subheader("📊 Combined Trust + Attack Probability (Dual Axis)")

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(df_dev.index, df_dev["trust_smooth"], color=PRIMARY, linewidth=2.4)
    ax1.set_ylabel("Trust", color=PRIMARY)

    ax2 = ax1.twinx()
    ax2.plot(df_dev.index, df_dev["attack_smooth"], color=DANGER, linewidth=2.4)
    ax2.set_ylabel("Attack Probability", color=DANGER)

    ax1.set_xlabel("Packet Index")
    ax1.grid(alpha=0.25)

    st.pyplot(fig)

# =====================================================
# TAB 3 — HEATMAP
# =====================================================
with tabs[3]:
    st.subheader("📊 QoS / Trust / Attack Correlation")
    corr_cols = [
        "qos_delay", "qos_jitter", "qos_throughput",
        "qos_bandwidth", "qos_packet_size", "qos_frame_size",
        "trust_score", "attack_prob"
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df_dev[corr_cols].corr(), annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

# =====================================================
# TAB 4 — RAW QOS TABLE
# =====================================================
with tabs[4]:
    st.subheader("📄 Raw QoS + Trust + Attack")
    qos_cols = [
        "qos_delay", "qos_jitter", "qos_throughput",
        "qos_bandwidth", "qos_packet_size", "qos_frame_size"
    ]

    df_show = df_dev[qos_cols + ["trust_score", "attack_prob"]].copy()

    sort_col = st.selectbox("Sort by:", df_show.columns)
    sort_order = st.radio("Order:", ["Ascending", "Descending"])

    df_show = df_show.sort_values(sort_col, ascending=(sort_order == "Ascending"))

    st.dataframe(df_show.head(10), use_container_width=True)

# =====================================================
# TAB 5 — TRUST VS SELECTED QOS
# =====================================================
with tabs[5]:
    st.subheader(f"🔍 Trust Score vs {metric_label}")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=df_dev,
        x="trust_score",
        y=metric_col,
        s=60,
        color=PRIMARY,
        alpha=0.7
    )
    ax.grid(alpha=0.25)
    st.pyplot(fig)

# =====================================================
# TAB 6 — THREAT GAUGE
# =====================================================
with tabs[6]:
    st.subheader("🛡 Threat Level Indicator")

    current_attack = float(df_dev["attack_prob"].iloc[-1])
    score = int(current_attack * 100)

    color = "#00cc44" if score < 30 else "#ffcc00" if score < 60 else "#cc0000"
    status = "SAFE" if score < 30 else "CAUTION" if score < 60 else "DANGER"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")

    ax.pie([1], colors=["#e6e6e6"])
    ax.pie([current_attack, 1 - current_attack], colors=[color, "white"], startangle=180)

    plt.text(0, -0.1, f"{score}%", ha="center", fontsize=22, weight="bold")
    plt.text(0, -0.3, status, ha="center", fontsize=18)

    st.pyplot(fig)

# =====================================================
# TAB 7 — DEVICE RISK RANKING
# =====================================================
with tabs[7]:
    st.subheader("📊 Device Risk Ranking")

    risk_table = df.groupby(["device", "device_label"]).agg({
        "attack_prob": "mean",
        "trust_score": "mean"
    }).reset_index()

    risk_table["risk_score"] = (
        (1 - risk_table["trust_score"]) * 0.6 +
        risk_table["attack_prob"] * 0.4
    )

    risk_table = risk_table.sort_values("risk_score", ascending=False)

    st.dataframe(risk_table, use_container_width=True)

# =====================================================
# TAB 8 — SMART HOME ATTACK OVERVIEW (ALL DEVICES)
# =====================================================
with tabs[8]:
    st.subheader("🏠 Smart Home Attack Overview — All Devices")

    device_summary = df.groupby(["device", "device_label"]).agg({
        "attack_prob": "mean",
        "trust_score": "mean",
        "qos_delay": "mean",
        "qos_jitter": "mean",
        "qos_throughput": "mean"
    }).reset_index()

    device_summary["risk_score"] = (
        (1 - device_summary["trust_score"]) * 0.6 +
        device_summary["attack_prob"] * 0.4
    )

    def risk_label(x):
        if x >= 0.6:
            return "DANGER"
        elif x >= 0.3:
            return "CAUTION"
        else:
            return "SAFE"

    device_summary["Risk Level"] = device_summary["risk_score"].apply(risk_label)

    color_map = {
        "DANGER": "#C00000",
        "CAUTION": "#FFCC00",
        "SAFE": "#00CC66"
    }
    device_summary["Color"] = device_summary["Risk Level"].map(color_map)

    # Likely attack type inference
    thr_q75 = df["qos_throughput"].quantile(0.75)
    jit_q85 = df["qos_jitter"].quantile(0.85)
    del_q85 = df["qos_delay"].quantile(0.85)
    thr_q30 = df["qos_throughput"].quantile(0.30)
    del_q70 = df["qos_delay"].quantile(0.70)

    def infer_attack_type(row):
        ap = row["attack_prob"]
        delay = row["qos_delay"]
        jitter = row["qos_jitter"]
        thr = row["qos_throughput"]
        trust = row["trust_score"]
        risk = row["risk_score"]

        if ap > 0.6 and thr > thr_q75:
            return "DDoS Attack"
        if jitter > jit_q85 and delay > del_q85:
            return "Data Manipulation / DoS"
        if trust < 0.4 and risk > 0.5:
            return "On-Off Attack"
        if (1 - thr) > 0.7 or ap > 0.7:
            return "Sinkhole Attack"
        if 0.3 < ap < 0.6 and trust < 0.7:
            return "Selective Forwarding"
        if delay > del_q70 and thr < thr_q30:
            return "Resource Exhaustion"
        return "Benign / Unknown"

    device_summary["Likely Attack Type"] = device_summary.apply(infer_attack_type, axis=1)

    device_summary = device_summary.sort_values("risk_score", ascending=False)

    st.markdown("### 📊 Device-Level Attack Risk & Likely Attack Type")
    st.dataframe(device_summary[[
        "device_label",
        "attack_prob",
        "trust_score",
        "risk_score",
        "Risk Level",
        "Likely Attack Type"
    ]], use_container_width=True)

    st.markdown("### 📈 Device Risk Bar Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(
        device_summary["device_label"],
        device_summary["risk_score"],
        color=device_summary["Color"]
    )
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Device")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    dangerous = device_summary[device_summary["Risk Level"] == "DANGER"][["device_label", "Likely Attack Type"]]

    if not dangerous.empty:
        st.markdown("### 🚨 Devices in DANGER State:")
        for _, row in dangerous.iterrows():
            st.markdown(f"- **{row['device_label']}** → `{row['Likely Attack Type']}`")
    else:
        st.markdown("### 🟢 No devices currently in DANGER state.")

# =====================================================
# TAB 9 — RISK RADAR CHART
# =====================================================
with tabs[9]:
    st.subheader(f"🕸 Risk Radar Chart — {device_sel_label}")

    radar_cols = [
        "qos_delay", "qos_jitter", "qos_throughput",
        "qos_bandwidth", "trust_score", "attack_prob"
    ]

    radar_values = df_dev[radar_cols].mean()
    vals = radar_values.values

    if (vals.max() - vals.min()) > 0:
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        vals_norm = np.zeros_like(vals)

    labels = ["Delay", "Jitter", "Throughput", "Bandwidth", "Trust", "Attack"]
    values = vals_norm.tolist() + [vals_norm.tolist()[0]]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += [angles[0]]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, values, linewidth=2, color=PRIMARY)
    ax.fill(angles, values, color=PRIMARY, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    ax.grid(alpha=0.3)
    st.pyplot(fig)

# =====================================================
# TAB 10 — ATTACK TYPE PREDICTION (RULE-BASED, DEVICE)
# =====================================================
with tabs[10]:
    st.subheader("🧠 Predicted Attack Type (Rule-Based, Selected Device)")

    avg_delay = df_dev["qos_delay"].mean()
    avg_jitter = df_dev["qos_jitter"].mean()
    avg_throughput = df_dev["qos_throughput"].mean()
    avg_bandwidth = df_dev["qos_bandwidth"].mean()
    avg_pkt = df_dev["qos_packet_size"].mean()
    avg_frame = df_dev["qos_frame_size"].mean()
    attack_prob_mean = df_dev["attack_prob"].mean()
    trust_avg = df_dev["trust_score"].mean()

    attack_type = "Benign / No Attack"
    explanation = ""

    thr_q75 = df_dev["qos_throughput"].quantile(0.75)
    jit_q85 = df_dev["qos_jitter"].quantile(0.85)
    del_q85 = df_dev["qos_delay"].quantile(0.85)
    pkt_q90 = df_dev["qos_packet_size"].quantile(0.90)

    if attack_prob_mean > 0.60 and avg_throughput > thr_q75:
        attack_type = "DDoS / Flooding Attack"
        explanation = "- High throughput\n- High attack probability\n- Flooding behaviour\n"
    elif avg_jitter > jit_q85 and avg_delay > del_q85:
        attack_type = "Congestion / DoS"
        explanation = "- Delay & jitter spikes\n- Congested behaviour\n"
    elif "dest_port" in df_dev.columns and df_dev["dest_port"].nunique() > 50:
        attack_type = "Port Scanning"
        explanation = "- Many destination ports\n- Possible reconnaissance\n"
    elif avg_pkt > pkt_q90:
        attack_type = "Data Exfiltration"
        explanation = "- Large packet sizes\n- Suspicious outbound traffic\n"
    elif "ttl" in df_dev.columns and df_dev["ttl"].nunique() > 10:
        attack_type = "Spoofing / Injection Attack"
        explanation = "- Abnormal TTL variation\n- Crafted packet signs\n"

    st.markdown(f"### 🔍 **Predicted Attack Type (Device):** `{attack_type}`")
    st.markdown("#### 📝 Reasoning:")
    st.write(explanation if explanation else "Normal / benign behaviour.")

    summary = {
        "Average Delay": avg_delay,
        "Average Jitter": avg_jitter,
        "Average Throughput": avg_throughput,
        "Average Bandwidth": avg_bandwidth,
        "Average Packet Size": avg_pkt,
        "Average Frame Size": avg_frame,
        "Average Trust Score": trust_avg,
        "Average Attack Probability": attack_prob_mean
    }

    st.subheader("📋 Behaviour Summary")
    st.table(summary)

# =====================================================
# TAB 11 — ATTACK TYPE SIGNATURES (QOS TABLE)
# =====================================================
with tabs[11]:
    st.subheader("🧩 Attack Type Signatures Based on QoS Behaviour")

    avg_delay = df_dev["qos_delay"].mean()
    avg_jitter = df_dev["qos_jitter"].mean()
    avg_throughput = df_dev["qos_throughput"].mean()
    attack_prob_mean = df_dev["attack_prob"].mean()
    packet_loss_proxy = attack_prob_mean * 0.7

    attack_table = pd.DataFrame({
        "Attack Type": [
            "DDoS Attack",
            "Data Manipulation",
            "On-Off Attack",
            "Sinkhole Attack",
            "Selective Forwarding",
            "Resource Exhaustion"
        ],
        "QoS Indicators": [
            f"↑ Throughput ({avg_throughput:.2f}), ↑ Loss Proxy ({packet_loss_proxy:.2f}), ↑ Delay ({avg_delay:.2f})",
            f"↑ Jitter ({avg_jitter:.2f}), ↑ Resp Time variance, ↓ Throughput consistency",
            "Fluctuating QoS, fluctuating Trust Score",
            f"↑ Severe loss proxy ({packet_loss_proxy:.2f}), ↓ Throughput ({avg_throughput:.2f})",
            "Structured packet loss patterns",
            f"↓ Response Time, ↑ Delay ({avg_delay:.2f}), ↓ Throughput ({avg_throughput:.2f})"
        ],
        "Behaviour Signature": [
            "Sudden traffic surge + device overload",
            "Irregular transmission & unstable flows",
            "Cyclic normal → malicious → normal pattern",
            "Data diverted or not delivered",
            "Specific packets dropped intentionally",
            "Slow, gradual degradation over time"
        ]
    })

    st.dataframe(attack_table, use_container_width=True)

# =====================================================
# TAB 12 — MOST LIKELY ATTACK TYPE (RANKING, DEVICE)
# =====================================================
with tabs[12]:
    st.subheader("🏆 Most Likely Attack Type (QoS + Trust Ranking — Selected Device)")

    def safe_norm(series):
        if series.max() - series.min() == 0:
            return np.zeros_like(series)
        return (series - series.min()) / (series.max() - series.min())

    delay_norm = safe_norm(df_dev["qos_delay"])
    jitter_norm = safe_norm(df_dev["qos_jitter"])
    thr_norm = safe_norm(df_dev["qos_throughput"])
    bw_norm = safe_norm(df_dev["qos_bandwidth"])
    trust_norm = safe_norm(df_dev["trust_score"])
    attack_norm = safe_norm(df_dev["attack_prob"])

    d = delay_norm.mean()
    j = jitter_norm.mean()
    t = thr_norm.mean()
    b = bw_norm.mean()
    tr = trust_norm.mean()
    ap = attack_norm.mean()

    ranking = []

    ddos_score = 0.4 * t + 0.3 * ap + 0.3 * d
    ranking.append(("DDoS Attack", ddos_score,
                    "High throughput + high anomaly score + increased delay"))

    data_manip_score = 0.5 * j + 0.3 * (1 - tr) + 0.2 * (1 - b)
    ranking.append(("Data Manipulation", data_manip_score,
                    "Jitter spikes and unstable traffic behaviour"))

    trust_std = trust_norm.std()
    thr_std = thr_norm.std()
    onoff_score = 0.5 * trust_std + 0.5 * thr_std
    ranking.append(("On-Off Attack", onoff_score,
                    "Cyclic changes in trust and traffic volume"))

    sinkhole_score = 0.6 * (1 - t) + 0.4 * ap
    ranking.append(("Sinkhole Attack", sinkhole_score,
                    "Traffic not reaching destination (effective loss + low throughput)"))

    sel_fwd_score = 0.5 * ap + 0.5 * (1 - tr)
    ranking.append(("Selective Forwarding", sel_fwd_score,
                    "Non-random anomalies suggesting targeted drops"))

    resource_score = 0.4 * d + 0.3 * (1 - t) + 0.3 * (1 - tr)
    ranking.append(("Resource Exhaustion", resource_score,
                    "Gradual slowdown and degraded QoS over time"))

    rank_df = pd.DataFrame(ranking, columns=["Attack Type", "Score", "Explanation"])
    rank_df = rank_df.sort_values("Score", ascending=False)

    st.markdown("### 🔝 Ranked Attack Behaviours for This Device")
    st.dataframe(rank_df, use_container_width=True)

    top_attack = rank_df.iloc[0]["Attack Type"]
    st.markdown(f"### ✅ Most Likely Behaviour: **{top_attack}**")


# =====================================================
# TAB 11 — PACKETS PER SECOND (NORMAL VS ATTACK)
# =====================================================
with tabs[10]:
    st.subheader("📡 Packets per Second — Normal vs Attack Traffic")

    # --------------------------------------------
    # DEVICE SELECTION
    # --------------------------------------------
    device_list = sorted(df["device_label"].unique())
    selected_device = st.selectbox("Select Device:", device_list)

    df_dev_packets = df[df["device_label"] == selected_device].copy()
    df_dev_packets = df_dev_packets.sort_values("epoch_timestamp").reset_index(drop=True)

    # --------------------------------------------
    # REAL PPS CALCULATION
    # --------------------------------------------
    if "inter_arrival_time" in df_dev_packets.columns:
        df_dev_packets["pps"] = 1 / df_dev_packets["inter_arrival_time"].replace(0, np.nan)
        df_dev_packets["pps"] = df_dev_packets["pps"].fillna(df_dev_packets["pps"].median())
    else:
        st.error("inter_arrival_time missing in dataset")
        st.stop()

    # ============================================================
    # NORMAL TRAFFIC (CLEANED & CROPPED FOR BETTER VISUAL QUALITY)
    # ============================================================
    st.markdown("### 🟢 Normal Traffic Pattern")

    # Remove extremely low PPS values (flat lines)
    normal_clean = df_dev_packets[df_dev_packets["pps"] > df_dev_packets["pps"].quantile(0.10)]

    # Smooth curve
    pps_normal = normal_clean["pps"].rolling(10).mean().dropna()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(pps_normal.index, pps_normal.values, color="black", linewidth=1.8)

    # Style: clean white background, thin gray grid
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(color="lightgray", linestyle="-", linewidth=0.5, alpha=0.4)

    ax.set_title(f"Normal Traffic: Packets per Second — {selected_device}", fontsize=18)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Packets / sec")

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # ATTACK TRAFFIC (ZOOMED & CLEANED)
    # ============================================================
    st.markdown("### 🔴 Attack Traffic Pattern")

    attack_df = df_dev_packets[df_dev_packets["attack_prob"] > 0.60]

    if len(attack_df) > 5:
        # remove low PPS noise and keep only active spikes
        attack_clean = attack_df[attack_df["pps"] > attack_df["pps"].quantile(0.30)]
        pps_attack = attack_clean["pps"].rolling(5).mean().dropna()
        title = f"Packets During Attack — {selected_device}"
    else:
        # fallback synthetic attack
        pps_attack = pd.Series(np.random.randint(12000, 20000, size=20))
        title = f"Packets During Synthetic Attack — {selected_device}"

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(pps_attack.index, pps_attack.values, color="black", linewidth=1.8)

    ax2.set_facecolor("white")
    fig2.patch.set_facecolor("white")
    ax2.grid(color="lightgray", linestyle="-", linewidth=0.5, alpha=0.4)

    ax2.set_title(title, fontsize=18)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Packets / sec")

    st.pyplot(fig2)

    st.info("""
    **How this works:**
    • PPS = 1 / inter_arrival_time  
    • Normal traffic is cleaned (removes long zero regions)  
    • Attack traffic is zoomed to active attack windows  
    • Produces clean visual graphs similar to academic papers  
    """)
