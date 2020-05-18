# -*- coding: utf-8 -*-
"""
Created on Mon May 18 22:48:26 2020

@author: boris
"""


#3-classification: 
#1- logistic regression

#inport the needed libraries
import numpy as np #to make mathematics computations
import matplotlib.pyplot as plt #used to build plots
import pandas as pd #to import and manage datasets


#import the dataset
dataset  = pd.read_csv("Social_Network_Ads.csv")


#create the matrix of the indepedant variables
"""
as the user id and their sex have no incidence on the decision of making a purchase
we can easily ignore the user id and the sex in our independentvariables
"""
X = dataset.iloc[:, [2,3]].values #do not  take the last column
#create the matrix of the depedant variable we want to predict
y = dataset.iloc[:,-1].values 


#no missing data 




#manage categorical variables
"""no categorical data to transform. 
NB: the dependent var is a categorical var, but it is already given in a binary 
way
"""

#divide the dataset between training set and test set 
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Feature scaling : should be applied here to put the variables in the same scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#♣building the logictic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

