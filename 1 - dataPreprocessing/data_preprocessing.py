# -*- coding: utf-8 -*-

"""
Created on Mon Apr 27 21:29:15 2020

@author: boris
"""

#Data Preprocessing

#inport the needed libraries
import numpy as np #to make mathematics computations
import matplotlib.pyplot as plt #used to build plots
import pandas as pd #to import and manage datasets


#import the dataset
dataset  = pd.read_csv("Data.csv")


#create the matrix of the indepedant variables
X = dataset.iloc[:,:-1].values #do not  take the last column
#create the matrix of the depedant variable we want to predict
Y = dataset.iloc[:,-1].values 


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
Y = labelEncoder_y.fit_transform(Y)
