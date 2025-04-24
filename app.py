import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Malaria Mutation Prediction", layout="wide")
st.title("🦟 Malaria Mutation Prediction Dashboard")

# Load prediction data
st.header("📊 Prediction Results")
try:
    with open("prediction.json") as f:
        data = json.load(f)
    df_pred = pd.DataFrame(data)
    st.dataframe(df_pred)
except Exception as e:
    st.error(f"Error loading prediction.json: {e}")

# Load and plot LSTM loss curve
st.header("📉 LSTM Training Loss")
try:
    loss_df = pd.read_csv("LSTM_loss.csv")
    st.write("CSV columns:", loss_df.columns.tolist())
    plt.figure(figsize=(10, 4))
    plt.plot(loss_df.iloc[:, 0], label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    st.pyplot(plt)
except Exception as e:
    st.error(f"Error loading LSTM_loss.csv: {e}")
