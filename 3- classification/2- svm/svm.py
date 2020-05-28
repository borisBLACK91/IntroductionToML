# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:55:47 2020

@author: boris
"""


#3-classification: 
#2- svm : Support Vector Margin
"""
model based on the support vector of each category. 
A support vector is a point of one category wich the nearest of the other category, 
meaning the point that could easily be confused as a point of the other category! 
So the 2 points of the 2 category matching that conditons are the support vectors.
"""

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
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

#making prediction with our model
y_pred = classifier.predict(X_test)


"""
confusion matrix to get the table of good predictions agains t wrong one
helpful to get the model fault positives, fault negatives, and accuracy
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#visualize results
from matplotlib.colors import ListedColormap 
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))                              
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X1.min(), X2.max())
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Resultats du traininig set')
plt.xlabel('Age')
plt.ylabel('Salaire Estimé')
plt.legend()
plt.show()





