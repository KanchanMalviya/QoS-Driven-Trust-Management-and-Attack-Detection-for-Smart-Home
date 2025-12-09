import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# ============================================
# CONFIG
# ============================================

DATA_PATH = "/Users/kanchan/Desktop/Fall-2025/IOT/CICIOT2022.parquet"

RANDOM_STATE = 42
ANOMALY_FRACTION = 0.05

HIGH_TRUST_THR   = 0.7
MEDIUM_TRUST_THR = 0.4


# ============================================
# 1. LOADING DATA
# ============================================

def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "epoch_timestamp" in df.columns:
        df["epoch_timestamp"] = pd.to_numeric(df["epoch_timestamp"], errors="coerce")

    return df


# ============================================
# 2. WEAK LABELING USING ISOLATION FOREST
# ============================================

def generate_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    X_raw = df[num_cols].fillna(0.0).values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)

    iso = IsolationForest(
        n_estimators=300,
        contamination=ANOMALY_FRACTION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    iso_pred = iso.fit_predict(X)
    df["weak_label"] = iso_pred
    df["label"] = df["weak_label"].replace({1: 0, -1: 1})

    print("Weak-label attack ratio:", df["label"].mean().round(4))
    return df


# ============================================
# 3. QOS METRIC CALCULATION
# ============================================

def compute_qos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["qos_packet_size"] = df["pck_size"]
    df["qos_frame_size"] = df["ethernet_frame_size"]
    df["qos_delay"] = df["inter_arrival_time"]
    df["qos_jitter"] = df["inter_arrival_time"].rolling(5).std().fillna(0.0)

    df["qos_throughput"] = df["total_length"] / (
        df["time_since_previously_displayed_frame"].replace(0, np.nan)
    )
    df["qos_throughput"] = df["qos_throughput"].fillna(df["qos_throughput"].median())

    df["qos_bandwidth"] = df["pck_size"].rolling(10).max().fillna(df["pck_size"])

    return df


# ============================================
# 4. TRUST COMPUTATION (FIXED, ATTACK-ALIGNED)
# ============================================

def compute_trust(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    qos_cols = [
        "qos_packet_size",
        "qos_frame_size",
        "qos_throughput",
        "qos_delay",
        "qos_jitter",
        "qos_bandwidth",
    ]

    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(
        scaler.fit_transform(df[qos_cols]),
        columns=[f"norm_{c}" for c in qos_cols],
        index=df.index
    )
    df = pd.concat([df, df_norm], axis=1)

    # FIX: metrics aligned with anomaly behaviour
    df["t_packet"] = df["norm_qos_packet_size"]
    df["t_frame"]  = df["norm_qos_frame_size"]
    df["t_thr"]    = 1 - df["norm_qos_throughput"]
    df["t_bw"]     = 1 - df["norm_qos_bandwidth"]
    df["t_delay"]  = 1 - df["norm_qos_delay"]
    df["t_jitter"] = 1 - df["norm_qos_jitter"]

    W_PACKET = 0.15
    W_FRAME  = 0.10
    W_THR    = 0.25
    W_BW     = 0.15
    W_DELAY  = 0.20
    W_JITT   = 0.15

    df["trust_score"] = (
        W_PACKET * df["t_packet"] +
        W_FRAME  * df["t_frame"]  +
        W_THR    * df["t_thr"]    +
        W_BW     * df["t_bw"]     +
        W_DELAY  * df["t_delay"]  +
        W_JITT   * df["t_jitter"]
    )

    return df


# ============================================
# 5. BUILD DATASET FOR RANDOM FOREST
# ============================================

def build_supervised_dataset(df: pd.DataFrame):
    df = df.copy()

    for col in ["device", "global_category"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    feature_cols = [
        "qos_packet_size", "qos_frame_size", "qos_throughput",
        "qos_delay", "qos_jitter", "qos_bandwidth",
        "trust_score",
        "sum_et", "min_et", "max_et", "med_et", "average_et",
        "skew_et", "kurt_et", "var", "iqr",
        "sum_e", "min_e", "max_e", "med", "average",
        "skew_e", "kurt_e", "var_e", "iqr_e",
        "device", "global_category"
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values
    y = df["label"].values

    return X, y, feature_cols


# ============================================
# 6. TRAIN SUPERVISED MODEL WITH BALANCED DATA
# ============================================

def balance_data(df):
    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]

    # Oversample attacks, undersample benign
    df1_over = resample(df1, replace=True, n_samples=20000, random_state=42)
    df0_under = resample(df0, replace=False, n_samples=20000, random_state=42)

    df_bal = pd.concat([df0_under, df1_over]).sample(frac=1, random_state=42)
    print("Balanced dataset → Benign:", len(df0_under), " Attack:", len(df1_over))
    return df_bal


def train_supervised_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    print("\n=== Supervised Model (Balanced) ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(confusion_matrix(y_test, y_pred))

    return scaler, clf


# ============================================
# 7. DECISION ENGINE
# ============================================

def decide_action(trust_score: float, attack_prob: float) -> str:
    if attack_prob >= 0.8:
        return "BLOCK / REMOVE"
    elif attack_prob >= 0.4:
        return "ISOLATE DEVICE"

    if trust_score >= HIGH_TRUST_THR:
        return "GRANT ACCESS"
    elif trust_score >= MEDIUM_TRUST_THR:
        return "MONITOR"
    else:
        return "ISOLATE DEVICE"


# ============================================
# 8. APPLY DECISIONS TO DATA
# ============================================

def apply_decisions(df, scaler, clf, feature_cols):
    df = df.copy()

    for col in ["device", "global_category"]:
        if col in df.columns and str(df[col].dtype) == "category":
            df[col] = df[col].cat.codes

    X_full = df[feature_cols].fillna(0).values
    X_full_s = scaler.transform(X_full)

    df["attack_prob"] = clf.predict_proba(X_full_s)[:, 1]

    df["action"] = [
        decide_action(t, p) for t, p in zip(df["trust_score"], df["attack_prob"])
    ]

    return df


# ============================================
# 9. MAIN PIPELINE
# ============================================

def main():

    print("[1] Loading data…")
    df = load_data(DATA_PATH)

    print("[2] Weak labeling…")
    df = generate_weak_labels(df)

    print("[3] QoS metrics…")
    df = compute_qos(df)

    print("[4] Trust calculation…")
    df = compute_trust(df)

    print("[5] Balancing dataset…")
    df_bal = balance_data(df)

    print("[6] Building ML dataset…")
    X, y, feature_cols = build_supervised_dataset(df_bal)

    print("[7] Training model…")
    scaler, clf = train_supervised_model(X, y)

    print("[8] Applying decisions on FULL data…")
    df_full = apply_decisions(df, scaler, clf, feature_cols)

    df_full.to_csv("full_pipeline_balanced_output.csv", index=False)
    print("\nSaved output → full_pipeline_balanced_output.csv")

    print("\n=== SAMPLE DECISIONS ===")
    print(df_full.sample(10)[["trust_score", "attack_prob", "action"]])

    return df_full


if __name__ == "__main__":
    df_out = main()
