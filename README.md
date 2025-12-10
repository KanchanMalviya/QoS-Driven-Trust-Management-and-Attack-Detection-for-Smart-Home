# QoS-Driven-Trust-Management-and-Attack-Detection-for-Smart-Home
##Overview
This project implements a trust-enhanced anomaly detection system for smart-home IoT security using the CICIoT2022 dataset.
The system computes QoS-derived trust scores, generates weak labels using Isolation Forest, and then trains a Random Forest classifier to predict attack probability.
A full Streamlit dashboard visualizes trust trends, QoS behavior, device-level risk, likely attack type, and normal vs attack traffic patterns—making this system suitable for real-world smart home monitoring.

## Key Features
📡 ## QoS Feature Extraction
Delay, jitter, throughput, bandwidth, packet size, frame size
🔐 **Trust Score Computation**
Weighted QoS-based behavioral trust with smoothing
🧪 **Weak Labeling (Isolation Forest)**
Automatic anomaly labeling from raw packet data
🌲 **Random Forest ML Model**
Trains with QoS only and QoS + Trust for comparison
🎯 **Decision Engine**
Determines: Grant Access, Monitor, Isolate, or Block
📊 **Real-Time Streamlit Dashboard**
Trust over time
Attack probability
Combined trust + probability
Device risk ranking
Most likely attack type
QoS heatmaps
Normal vs attack traffic behavior
🔍 **End-to-End Smart Home Security Pipeline**

**This architecture includes:**
Raw IoT traffic loading
QoS metric computation
Trust scoring module
Weak labeling via Isolation Forest
Balanced training data generation
Random Forest classifier
Prediction: attack probability
Decision engine
Visualization dashboard


**Project Structure**
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




| Component     | Technology                     |
| ------------- | ------------------------------ |
| ML Model      | Random Forest (scikit-learn)   |
| Weak Labels   | Isolation Forest               |
| Dashboard     | Streamlit                      |
| Data Source   | CICIoT2022 IoT Traffic Dataset |
| Core Language | Python 3.x                     |
| Visualization | Matplotlib, Seaborn            |



**Requirements**
Python 3.9+
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
pyarrow (for parquet dataset)


pip install -r requirements.txt

**How to Run the Project**
Run the Trust + ML Pipeline
This generates full_pipeline_balanced_output.csv.
python core_trust.py

Launch the Streamlit Dashboard
streamlit run dashboard.py

**Model Summary**
Baseline Random Forest (QoS only): ~84% accuracy
Enhanced Random Forest (QoS + Trust): ~98.2% accuracy
Trust score greatly reduces false positives
Effective detection of:
DDoS
Data manipulation
On-off attacks
Sinkhole attacks
Selective forwarding
Resource exhaustion



