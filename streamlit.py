import streamlit as st
import joblib

model = joblib.load("cell_anomaly_classifier.pkl.pkl")

st.title("cell_anomaly_classifier")
st.write("Enter CCE congestion, PRB congestion, sinr, rsrp, handover success rate and drop rate")

# 3 inputs
charge_CCE = st.number_input("CCE congestion", min_value=0.0, step=1.0)
charge_PRB    = st.number_input("charge_PRB",        min_value=0.0, step=1.0)
rsrp     = st.number_input("rsrp", min_value=0.0, step=1.0)
sinr     = st.number_input("sinr", min_value=0.0, step=1.0)
ho_success_rate=st.number_input("ho_success_rate", min_value=0.0, step=1.0)
call_drop_rate_pct=st.number_input("call_drop_rate_pct", min_value=0.0, step=1.0)

if st.button("Predict"):
    try:
        X = [[charge_CCE, Traffic, NBR_UE]]
        prediction = model.predict(X)
        st.write(f"Predicted cell state: {float(prediction[0]):.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
      
