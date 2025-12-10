# QoS-Driven-Trust-Management-and-Attack-Detection-for-Smart-Home

##  Overview
This project implements a **trust-enhanced anomaly detection system** for smart-home IoT security using the **CICIoT2022 dataset**.

The system:
- Computes **QoS-derived trust scores**
- Generates **weak labels using Isolation Forest**
- Trains a **Random Forest classifier** to predict attack probability
- Provides a full **Streamlit dashboard** to visualize trust, QoS behavior, device risk, attack types, and real-time smart-home monitoring

---

## ⭐ Key Features

### 📡 QoS Feature Extraction  
- Delay  
- Jitter  
- Throughput  
- Bandwidth  
- Packet size  
- Frame size  

### 🔐 Trust Score Computation  
- Weighted QoS-based behavioral trust  
- λ-smoothing to reduce noise  

### 🧪 Weak Labeling (Isolation Forest)  
- Automatic anomaly detection from raw IoT traffic  

### 🌲 Random Forest ML Model  
- Baseline: QoS-only features  
- Enhanced: QoS + Trust Score  

### 🎯 Decision Engine  
- Grant Access  
- Monitor  
- Isolate  
- Block/Remove  

### 📊 Real-Time Streamlit Dashboard  
- Trust over time  
- Attack probability over time  
- Combined Trust + Attack Probability  
- Device risk ranking  
- Likely attack type  
- QoS correlation heatmaps  
- Normal vs attack traffic visualization  

### 🔍 End-to-End Smart Home Security Pipeline  
- Raw traffic loading  
- QoS metric computation  
- Trust scoring module  
- Weak labeling  
- Balanced training data generation  
- Random Forest classifier  
- Attack probability prediction  
- Decision engine  
- Real-time visualization dashboard  

---

## 📁 Project Structure
IOT_Trust_Final_Project/
│
├── core_trust.py                      # Main ML pipeline
├── dashboard.py                       # Streamlit dashboard
│
├── full_pipeline_balanced_output.csv  # Output used by dashboard
│
├── README.md                          # GitHub documentation
├── requirements.txt                   # Dependencies
│
└── images/  
      ├── system_block_diagram.png
      ├── trust_vs_attack.png
      └── rf_performance.png


---

## Tech Stack

| Component     | Technology                     |
|---------------|---------------------------------|
| ML Model      | Random Forest (scikit-learn)    |
| Weak Labels   | Isolation Forest                |
| Dashboard     | Streamlit                       |
| Data Source   | CICIoT2022 IoT Traffic Dataset  |
| Core Language | Python 3.x                      |
| Visualization | Matplotlib, Seaborn             |

---

##  Requirements

- Python 3.9+  
- pandas  
- numpy  
- scikit-learn  
- streamlit  
- matplotlib  
- seaborn  
- pyarrow (for parquet dataset)

## 🚀 How to Run the Project

### 1️⃣ Install all dependencies
```bash
pip install -r requirements.txt

python core_trust.py






## How to Run the Project
1. Install all dependencies:

```bash
pip install -r requirements.txt```
 
2. Run the Trust + ML Pipeline
Generates full_pipeline_balanced_output.csv
```bash python core_trust.py```
3. Launch the Streamlit Dashboard
```bash streamlit run dashboard.py```

##  License
For educational and research use only.

##  Acknowledgements
CICIoT2022 Dataset
Streamlit
scikit-learn
Matplotlib / Seaborn

