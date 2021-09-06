#Decison tree regression 
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Splitting the dataset into training set and test set
"""from sklearn.moldel_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                               random_state=0)"""
#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#Fitting Regression model to the dataset


#Predicting a new result
y_pred=regressor.predict(np.array([6.5]).reshape(1,1))

#Visualizing the model
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X),color="green")
plt.title("Truth of bluff(reg model)")
plt.xlabel("Posiltion salaries")
plt.ylabel("Salary")
plt.show() 

#Visualizing the Regression in  a higher way
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="green")
plt.title("Truth of bluff(reg model)")
plt.xlabel("Posiltion salaries")
plt.ylabel("Salary")
plt.show() 

