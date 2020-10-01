# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 21:04:19 2020

@author: Walter Stevens

Use of decision trees on the Iris dataset. 


"""
import pandas as pd
from pathlib import Path

# Importing the dataset
dataset = pd.read_csv(Path.cwd()/'Iris.csv')

import numpy as np
import pandas as pd
from sklearn import tree



#Initial Data discovery
no_Samples, no_Features = dataset.shape
#print(no_Samples)
#150
#print(no_Features)
#6
#print(dataset.Species.unique())
#['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']

#diction = {"Iris-setosa":0,"Iris-versicolor":1,'Iris-virginica':2}
#since the dataset contained the actual sub-species names e.g. Iris-setosa,
#I had to replace those categorical variables with  numeric ones (9,1,2)

#dataset.replace({"Species": diction},inplace = True)


s = dataset['Species']
y = s.values.tolist()
X = dataset

ydf = pd.DataFrame(y,columns = ['Species'])


X_feature_names = X.columns.tolist()
y_target_names = dataset.Species.unique().tolist()

X.drop('Species', 1, inplace = True)
X.drop('Id', 1, inplace = True)

#Splitting off 20% for the test data, leaving me with 80%
#for the training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, ydf,test_size=0.2, random_state=0)

#Now actually construct the decision tree:
    
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,ydf)

#Now to display the tree:
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(30,25))
a = plot_tree(clf, 
              feature_names=X_feature_names, 
              class_names=y_target_names, 
              filled=True, 
              rounded=True, 
              fontsize=20)

#Testing the accuracy
from sklearn.model_selection import cross_val_score
print("Cross-val Score")
a = cross_val_score(clf, X, ydf, cv=5)
print(a)




