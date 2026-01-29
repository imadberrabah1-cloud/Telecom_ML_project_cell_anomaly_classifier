import streamlit as st
import joblib

model = joblib.load("cell_anomaly_classifier.pkl")

st.title("Cell Anomaly Classifier")
st.write("Enter KPIs to predict the cell state")

charge_CCE = st.number_input("CCE congestion", min_value=0.0, value=50.0, step=1.0)
charge_PRB = st.number_input("PRB congestion", min_value=0.0, value=60.0, step=1.0)

rsrp = st.number_input("RSRP (dBm)", value=-110.0, step=1.0)      # can be negative
sinr = st.number_input("SINR (dB)", value=-3.0, step=0.5)         # can be negative

ho_success_rate = st.number_input("HO success rate (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
call_drop_rate_pct = st.number_input("Call drop rate (%)", min_value=0.0, value=2.0, step=0.1)

if st.button("Predict"):
    try:
        X = [[charge_CCE, charge_PRB, rsrp, sinr, ho_success_rate, call_drop_rate_pct]]
        prediction = model.predict(X)
        st.success(f"Predicted cell state: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")