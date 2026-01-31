 Cell Anomaly Classifier

A machine learning application to classify radio cell anomalies based on key RAN performance indicators (PRB, CCE, RSRP, SINR, handover success rate, call drop rate).

The project includes:
- A production-ready REST API using **FastAPI**
- A user-friendly demo interface using **Streamlit**
- Full containerization with **Docker**

---

Features

- Predicts cell status such as:
  - CONGESTED
  - BAD_HANDOVER
  - RADIO_CONDITION_ISSUE
  - GOOD_CELL
- Real-time inference via REST API
- Input validation with Pydantic
- Preloaded model for low latency
- Ready for local or cloud deployment

---

 Tech Stack

- **Python**
- **Scikit-learn**
- **FastAPI**
- **Streamlit**
- **Docker**
- **Pandas / NumPy**
- **Joblib**
- **Git & GitHub**

---

Project Structure

```text
cell_anomaly_classifier_project/
│
├── main.py                    # FastAPI backend (ML inference API)
├── streamlit.py               # Streamlit demo application
│
├── cell_anomaly_classifier.pkl  # Trained ML model
├── scaler.pkl                  # Feature scaler
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── .env                        # Environment variables
│
├── static/
│   └── index.html              # Simple frontend page
│
└── Cell_Anomaly_Classifier.ipynb # Model training & experimentation
