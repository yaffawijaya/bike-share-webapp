import streamlit as st
import pandas as pd
import numpy as np

# Accepting user data for predicting its Member Type
def accept_user_data():
	duration = st.text_input("Enter the Duration: ")
	start_station = st.text_input("Enter the start station number: ")
	end_station = st.text_input("Enter the end station number: ")
	user_prediction_data = np.array([duration,start_station,end_station]).reshape(1,-1)

	return user_prediction_data

# Loading the data for showing visualization of vehicals starting from various start locations on the world map.
@st.cache_data
def showMap():
	plotData = pd.read_csv("Trip history with locations.csv")
	Data = pd.DataFrame()
	Data['lat'] = plotData['lat']
	Data['lon'] = plotData['lon']

	return Data