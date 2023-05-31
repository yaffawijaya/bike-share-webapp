import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler  
import plotly.express as px
from model import *
from viz import *

@st.cache_data
def loadData():
	df = pd.read_csv('2010-capitalbikeshare-tripdata.csv')  
	return df

def main():
	st.title("Bike Trip History")
	st.text("Predict and understanding bicycles rent in near Washington DC")
	
	data = loadData()
	X_train, X_test, y_train, y_test, le = preprocessing(data)

	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Show and Describe Raw Data'):
		st.subheader("Raw Data:")	
		st.write(data)
		st.subheader("Describe:")
		st.write(data.describe())

	st.markdown("""---""")

	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["-","Decision Tree", "Neural Network"])

	if(choose_model == "Decision Tree"):
		score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
		st.subheader("Accuracy of Decision Tree model: ")
		st.write(score,"%")
		st.subheader("Decision Tree model report: ")
		st.write(report)

		try:
			st.subheader("Want to predict on your own Input?")
			st.text('💡Tips: Use the information reference to the dataset for the Member and Casual member class type output')
			if(st.checkbox('Show Input')):
				user_prediction_data = accept_user_data() 		
				pred = tree.predict(user_prediction_data)
				clss = le.inverse_transform(pred)
				st.write("The Predicted Class is: ", clss) # Inverse transform to get the original dependent value. 
		except:
			pass

	elif(choose_model == "Neural Network"):
		score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
		st.subheader("Accuracy of Neural Network model: ")
		st.write(score,"%")
		st.subheader("Neural Network model report: ")
		st.write(report)

		try:
			st.subheader("Want to predict on your own Input?")
			st.text('💡Tips: Use the information reference to the dataset for the Member and Casual member class type output')
			if(st.checkbox('Show Input')):
				user_prediction_data = accept_user_data()
				scaler = StandardScaler()  
				scaler.fit(X_train)  
				user_prediction_data = scaler.transform(user_prediction_data)	
				pred = clf.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
		except:
			pass

	# Visualization Section
	plotData = showMap()
	st.markdown("""---""")
	st.subheader("Bike Travel History data plotted-first few locations located near Washington DC")
	st.map(plotData, zoom = 5, use_container_width=True)


	choose_viz = st.sidebar.selectbox("Choose the Visualization",
		["-","Member Type","Start Points", "End Points"])
	
	st.markdown("""---""")
	if(choose_viz == "Member Type"):
		fig = px.histogram(data['Member type'], x ='Member type', title='Histogram of Member Type')
		st.plotly_chart(fig)
	elif(choose_viz == "Start Points"):
		fig = px.histogram(data['Start station'], x ='Start station', title='Histogram of Start Stations')
		st.plotly_chart(fig)
	elif(choose_viz == "End Points"):
		fig = px.histogram(data['End station'], x ='End station', title='Histogram of End Stations')
		st.plotly_chart(fig)

if __name__ == "__main__":
	main()
	st.write("Author: Kelompok 1 - PERANCANGAN APLIKASI SAINS DATA DS-45-03")