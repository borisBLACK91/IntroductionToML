# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:06:36 2020
@author: boris
"""

# -*- coding: utf-8 -*-


#inport the needed libraries
import numpy as np #to make mathematics computations
import matplotlib.pyplot as plt #used to build plots
import pandas as pd #to import and manage datasets


#import the dataset
dataset  = pd.read_csv("Position_Salaries.csv")


#create the matrix of the indepedant variables
X = dataset.iloc[:,1:2].values #do not  take the last column
#create the matrix of the depedant variable we want to predict
y = dataset.iloc[:,-1].values 


#manage missing data
#import the SimpleImputer class 
""" commmented as we dont have null values
from sklearn.impute import SimpleImputer
#instantiate an imputer object that will repalkce missing values
imputer = SimpleImputer(missing_values = np.nan , strategy = "mean")
#compute the mean of each column to replace the nan values
imputer.fit(X[:,1:3])
#replace the missing values with the mean of the  corresponding columns
X[:, 1:3] = imputer.transform(X[:, 1:3]) # 1:3 means coluns index : 1, 2 
"""


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
""" commented as we dont have categorical vars
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#we first transfrom countries values to num values
columnTransformer = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder ="passthrough")
#then we transform X with the values transformed
X = columnTransformer.fit_transform(X)
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
"""



#building the polynomial model
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)

#make new predictions using the regressor
#first using the test set X_test
y_pred = regressor.predict(X_test)

#predict other values taht are not in the test set
regressor.predict([[15]])


#plot the results
plt.scatter(X, y, color = "red")
plt.plot(X, regressor.predict(X_poly), color = "blue")
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()