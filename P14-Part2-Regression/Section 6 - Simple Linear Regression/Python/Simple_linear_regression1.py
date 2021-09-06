# Simple linear regression

#importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Importing the dataset
dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#Splitting the dataset into training set and test Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#Fiiting simplel linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set result
y_pred=regressor.predict(X_test)

#Accuracy score for training
from sklearn.metrics import accuracy_score
a_s1=regressor.score(X_train,y_train)

#Accuracy score for testing

a_s2=regressor.score(X_test,y_test)

#Visualizing the training set result
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title("salary vs Experience(training set)")
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()

#Visualizing the Test set result
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("salary vs Experience(Test set)")
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()











