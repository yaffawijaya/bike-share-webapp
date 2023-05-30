import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler  
import plotly.express as px
import google.auth
from google.cloud import storage

from model import *
from viz import *

def get_csv_gcs(bucket_name, file_name):
    csv_data = pd.read_csv('gs://' + bucket_name + '/' + file_name, encoding='utf-8')  
    # csv_data = pd.read_excel('gs://' + bucket_name + '/' + file_name, encoding='utf-8')    
    return csv_data

bucket_name = "mybucket-bikeshare"
file_name = "2010-capitalbikeshare-tripdata.csv"

@st.cache_data
def loadData(bucket_name, file_name):
	df = pd.read_csv('gs://' + bucket_name + '/' + file_name, encoding='utf-8')  
	return df

# Basic preprocessing required for all the models.  
def preprocessing(df):
	# Assign X and y
	X = df.iloc[:, [0, 3, 5]].values
	y = df.iloc[:, -1].values

	# X and y has Categorical data hence needs Encoding
	le = LabelEncoder()
	y = le.fit_transform(y.flatten())

	# 1. Splitting X,y into Train & Test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
	return X_train, X_test, y_train, y_test, le


def main():
	st.title("Prediction of Trip History Data")
	st.text("The implementation Machine Learning Classification Algorithms")
	bucket_name = "dataprep-staging-76a4eba9-abb7-41ce-9168-df23035f64aa/yaffazka@gmail.com/jobrun"
	file_name = "Join Table.csv"
	data = loadData(bucket_name,file_name)
	X_train, X_test, y_train, y_test, le = preprocessing(data)

	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Show Raw Data'):
		st.subheader("Showing raw data---->>>")	
		st.write(data.head(10))


	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["NONE","Decision Tree", "Neural Network", "K-Nearest Neighbours"])

	if(choose_model == "Decision Tree"):
		score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Decision Tree model is: ")
		st.write(score,"%")
		st.text("Report of Decision Tree model is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
				user_prediction_data = accept_user_data() 		
				pred = tree.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
		except:
			pass

	elif(choose_model == "Neural Network"):
		score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Neural Network model is: ")
		st.write(score,"%")
		st.text("Report of Neural Network model is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
				user_prediction_data = accept_user_data()
				scaler = StandardScaler()  
				scaler.fit(X_train)  
				user_prediction_data = scaler.transform(user_prediction_data)	
				pred = clf.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
		except:
			pass

	elif(choose_model == "K-Nearest Neighbours"):
		score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
		st.text("Accuracy of K-Nearest Neighbour model is: ")
		st.write(score,"%")
		st.text("Report of K-Nearest Neighbour model is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
				user_prediction_data = accept_user_data() 		
				pred = clf.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
		except:
			pass
	
	
	# Visualization Section
	plotData = showMap()
	st.subheader("Bike Travel History data plotted-first few locations located near Washington DC")
	st.map(plotData, zoom = 7, use_container_width=True)


	choose_viz = st.sidebar.selectbox("Choose the Visualization",
		["NONE","Total number of vehicles from various Starting Points", "Total number of vehicles from various End Points",
		"Count of each Member Type"])
	
	if(choose_viz == "Total number of vehicles from various Starting Points"):
		fig = px.histogram(data['Start station'], x ='Start station')
		st.plotly_chart(fig)
	elif(choose_viz == "Total number of vehicles from various End Points"):
		fig = px.histogram(data['End station'], x ='End station')
		st.plotly_chart(fig)
	elif(choose_viz == "Count of each Member Type"):
		fig = px.histogram(data['Member type'], x ='Member type')
		st.plotly_chart(fig)

	# plt.hist(data['Member type'], bins=5)
	# st.pyplot()
	#test

if __name__ == "__main__":
	main()