# Training Decission Tree for Classification
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

@st.cache_resource(experimental_allow_widgets=True)
def decisionTree(X_train, X_test, y_train, y_test):
	# Train the model
	tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
	tree.fit(X_train, y_train)
	y_pred = tree.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score, report, tree

# Training Neural Network for Classification.
@st.cache_resource(experimental_allow_widgets=True)
def neuralNet(X_train, X_test, y_train, y_test):
	# Scalling the data before feeding it to the Neural Network.
	scaler = StandardScaler()  
	scaler.fit(X_train)  
	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)
	# Instantiate the Classifier and fit the model.
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score1 = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)
	
	return score1, report, clf