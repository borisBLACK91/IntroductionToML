# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:15:56 2020

@author: boris
"""


#1-Data Preprocessing

#inport the needed libraries
import numpy as np #to make mathematics computations
import matplotlib.pyplot as plt #used to build plots
import pandas as pd #to import and manage datasets


#import the dataset
dataset  = pd.read_csv("50_Startups.csv")

#create the matrix of the indepedant variables
X = dataset.iloc[:,:-1].values #do not  take the last column
#create the matrix of the depedant variable we want to predict
y = dataset.iloc[:,-1].values 


#manage missing data
"""
NON MISSING DATA IN THIS DATA SET
#import the SimpleImputer class 
from sklearn.impute import SimpleImputer
#instantiate an imputer object that will repalkce missing values
imputer = SimpleImputer(missing_values = np.nan , strategy = "mean")
#compute the mean of each column to replace the nan values
imputer.fit(X[:,1:3])
#replace the missing values with the mean of the  corresponding columns
X[:, 1:3] = imputer.transform(X[:, 1:3]) # 1:3 means coluns index : 1, 2 
"""


#manage categorical variables
#the state (index = 3) is a categorical variable 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#we first transfrom countries values to num values
columnTransformer = ColumnTransformer([("Country", OneHotEncoder(), [3])], remainder ="passthrough")
#then we transform X with the values transformed
X = columnTransformer.fit_transform(X)
#as th dependent variable (the startups profit) is not a categorical var we dont need the label encoder
#labelEncoder_y = LabelEncoder()
#y = labelEncoder_y.fit_transform(y)


#let's remove one dummy variables to keep independence of variable property 
X = X[:, 1:]


#divide the dataset between training set and test set 
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
So we build the ML model based on the training set (X_train, y_train) ans then 
we estimate the model precision using the test set (X_test , y_test)
"""

#Feature scaling 
#no need of feature scaling in Multiple Linear regression as the coefs of each var could adapt the the scales
"""putting all variables on the same scale , 
to avoid huge values variable to crush small values varibales and compromise
our model
""" 
#from sklearn.preprocessing import StandardScaler
"""
how does the scaling works : 2 main formukas 
-standardisation: 
    Xstand = [X - mean(X)]/standardDeviation(X)
or
-normalisation 
    Xnorm = [X - min(X)]/[max(X) - min(X)]   
"""
"""sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""


#building the Multiple Linear Regression model (MLR Model)
from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#make new predictions using the regressor
#first using the test set X_test
y_pred = regressor.predict(X_test)

#predict other values taht are not in the test set
#regressor.predict([[15]])
"""below the prediction of the profit of a startup in Newyork (1, 0 as for the categorical
variable transformation), with RD Spend = 130000, Admin = 140000, Marketing = 300000
"""
regressor.predict(np.array([[1, 0, 130000, 140000, 300000]]))