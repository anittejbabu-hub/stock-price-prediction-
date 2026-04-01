import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("📊 Advanced Stock Prediction Dashboard")

# Company selection
company = st.selectbox(
    "Select Company",
    ["TSLA", "AAPL", "AMZN", "GOOGL"]
)

# Download data
data = yf.download(company, period="1y")

# Show raw data
st.subheader("📄 Stock Data")
st.write(data.tail())

# Plot chart
st.subheader("📈 Stock Price Chart")
fig, ax = plt.subplots()
ax.plot(data['Close'])
st.pyplot(fig)

# Current price
current_price = float(data['Close'].iloc[-1])
st.write(f"💰 Current Price: {current_price:.2f}")

# Prediction
st.subheader("🔮 Predict Future Price")

input_price = st.number_input("Enter price for prediction", value=float(current_price))

if st.button("Predict"):
    prediction = model.predict([[input_price]])
    st.success(f"📈 Predicted Price after 30 days: {prediction[0]:.2f}")