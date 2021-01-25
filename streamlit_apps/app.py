import os
import streamlit as st
from datetime import date, timedelta
from pandas_ml_quant import pd
from importlib import import_module

import pandas_quant_data_provider
from streamlit_apps.config import Config

st.title("hello world")


symbol = st.text_input("Symbol", "SPY")
start_date = st.date_input("Start Date", date(1998, 2, 1))

df = pd.fetch_yahoo(symbol)[start_date:]
st.write(df.tail())


days = st.slider("# of days", 20, 5000, 90, 5)
data = df["Close"].iloc[-days:]
st.line_chart(data)


model_definition_file = st.selectbox("Model", [m for m in os.listdir(Config.model_directory) if m.endswith(".py")])
st.write("selected model", model_definition_file)

# TODO we have a model file (ending with .py) and eventually a drained model (ending with .model)
#  if a trained model for the current symbol and the selected model definition file exists
#  then show an apply and train button
#  else only show a train button

"""
Asset
  Select one asset
  
Chart
 show a chart of the current asset and provide a slider how many days to show (stepsize 5)
 
Model
  Show a list of available models
  Add a train model button and some possible inputs
    - eventually add a progress bar 
  Show chart of the result of the model
    - eventually add spinner
    
"""