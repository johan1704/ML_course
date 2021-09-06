#REVIEWING the Gradient Boosting or Ensemble Classifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
data=pd.read_csv("Churn_Modelling.csv")
X=data.iloc[:,3:13].values
y=data.iloc[:,13].values

#Let's perform Data exploratory
data.isna().sum()
"""
the data don't contain any missing values so we can start the feature engineering
Possibility and better to feature scaling data
"""
data.describe()
data.dtypes

#we need to take care of categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
ct = ColumnTransformer([("Geography,gender", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X).toarray()
X=X[:,1:]

#Divide our data into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Split the model
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

#Making the predictions
y_pred=classifier.predict(X_test)

#Check for the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Prediction of a single value
new_prediction=classifier.predict(sc_X.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
if new_prediction==0:
    print("the customer is going to leave the bank")
else:
    print("Staying")
new_prediction    
    