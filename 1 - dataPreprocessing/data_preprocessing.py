# -*- coding: utf-8 -*-

"""
Created on Mon Apr 27 21:29:15 2020

@author: boris
"""

#1-Data Preprocessing

#inport the needed libraries
import numpy as np #to make mathematics computations
import matplotlib.pyplot as plt #used to build plots
import pandas as pd #to import and manage datasets


#import the dataset
dataset  = pd.read_csv("Data.csv")


#create the matrix of the indepedant variables
X = dataset.iloc[:,:-1].values #do not  take the last column
#create the matrix of the depedant variable we want to predict
y = dataset.iloc[:,-1].values 


#manage missing data
#import the SimpleImputer class 
from sklearn.impute import SimpleImputer
#instantiate an imputer object that will repalkce missing values
imputer = SimpleImputer(missing_values = np.nan , strategy = "mean")
#compute the mean of each column to replace the nan values
imputer.fit(X[:,1:3])
#replace the missing values with the mean of the  corresponding columns
X[:, 1:3] = imputer.transform(X[:, 1:3]) # 1:3 means coluns index : 1, 2 



#manage categorical variables
"""
we need to encode categorical variables into numerical values 
to be able to use them inside mathematical equations. 
So in our dataset we need to do that for the
independant variable Country and the dependant 
variable Purchased 
-> we need to know if those cathegorical variable are 
nominal 
or 
ordinal
with dummy variable (onehot) (a new column for each values) when there no notion of orders -> country. 
for the var Purchased we will use 0 for no and 1 for yes. We can do this because it is the dependant var
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#we first transfrom countries values to num values
columnTransformer = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder ="passthrough")
#then we transform X with the values transformed
X = columnTransformer.fit_transform(X)
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)


#divide the dataset between training set and test set 
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
So we build the ML model based on the training set (X_train, y_train) ans then 
we estimate the model precision using the test set (X_test , y_test)
"""

#Feature scaling 
"""putting all variables on the same scale , 
to avoid huge values variable to crush small values varibales and compromise
our model
""" 
from sklearn.preprocessing import StandardScaler
"""
how does the scaling works : 2 main formukas 
-standardisation: 
    Xstand = [X - mean(X)]/standardDeviation(X)
or
-normalisation 
    Xnorm = [X - min(X)]/[max(X) - min(X)]   
"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


