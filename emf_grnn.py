# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 21:54:58 2021

@author: eshar
"""


#Libraries needed to run the tool
import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from neupy.algorithms import RBFKMeans
from neupy.algorithms import GRNN
import seaborn as sns
sns.set(style='darkgrid')

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'emf_data'
data_raw = pd.read_csv(file_name + '.csv', header=0, index_col=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data_raw.index), len(data_raw.columns)))
print("")

#Removing Letter
#data = data_raw.drop(['Letter'], axis=1)
data=data_raw
#Using Built in train test split function in sklearn
data_train, data_test = train_test_split(data, test_size=0.25)

#Hacking a scaling but keeping columns names since min_max_scaler does not return a dataframe
minval = data_train.min()
minmax = data_train.max() - data_train.min()
data_train_scaled = (data_train - minval) / minmax
data_test_scaled = (data_test - minval) / minmax

#Define X and Y
X_train = data_train_scaled.drop(['emf'], axis=1)
Y_train = data_train_scaled.emf
X_test = data_test_scaled.drop(['emf'], axis=1)
Y_test = data_test_scaled.emf


# RBF and kmeans clustering by class 
#Number of prototypes
#prototypes = int(input("Number of seed points:"))
prototypes = 20

#Finding cluster centers
df_cluster = X_train
df_cluster['emf']= Y_train #Reproduce original data but only with training values
rbfk_net = RBFKMeans(n_clusters=prototypes) #Chose number of clusters that you want
rbfk_net.train(df_cluster, epsilon=1e-5)
center = pd.DataFrame(rbfk_net.centers)

# Turn the centers into prototypes values needed
X_prototypes = center.iloc[:, 0:-1]
Y_prototypes = center.iloc[:, -1] #Y_prototypes is the last column of center since 'emf' is the last feature added to center.

#Train GRNN
GRNNet = GRNN(std=0.1) #Learn more at http://neupy.com/apidocs/neupy.algorithms.rbfn.grnn.html
GRNNet.train(X_prototypes, Y_prototypes, copy=True)

# Cross validataion
score = cross_val_score(GRNNet, X_train, Y_train, scoring='r2', cv=5)
print("")
print("Cross Validation: {0} (+/- {1})".format(score.mean().round(2), (score.std() * 2).round(2)))
print("")

#Prediction
Y_predict = GRNNet.predict(X_test) #for any prediction: GRNNet.predict([[0.5,0.1,0.2]])
print("Test: {0}".format((Y_test.values * minmax.emf + minval.emf).round(3)))
print("")
print("Predict: {0}".format((Y_predict * minmax.emf + minval.emf)[:,0].round(3)))
print("")
print("Accuracy: {0}".format(metrics.r2_score(Y_test, Y_predict).round(2)))
print("")
'''
#plot
A=np.array((Y_test.values * minmax.emf + minval.emf).round(3))
P=np.array((Y_predict * minmax.emf + minval.emf)[:,0].round(3))
AA=np.array((Y_train.values * minmax.emf + minval.emf).round(3))
#PP=np.array(GRNNet.predict(X_train))
print(A)
print(P)
print(AA)
#print(PP)
plt.scatter(P,A, color='tab:red',alpha=.5)
plt.show()
'''
#comparison
#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name_p = 'emf_data'
data_raw_test = pd.read_csv(file_name_p + '.csv', header=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data_raw.index), len(data_raw.columns)))
print("")
#Removing Letter
#data = data_raw.drop(['Letter'], axis=1)
data_p=data_raw_test
#Hacking a scaling but keeping columns names since min_max_scaler does not return a dataframe
minval_p= data_p.min()
minmax_p = data_p.max() - data_p.min()
data_t_scaled = (data_p - minval_p) / minmax_p

#data_test_scaled = (data_test - minval) / minmax

#Define X and Y
# X_train = data_train_scaled.drop(['emf'], axis=1)
# Y_train = data_train_scaled.emf
#data.drop(columns=['Letter', 'Grade'])
X_p = data_t_scaled.drop(columns=['emf','Time'])
Y_p = data_t_scaled.emf
#Prediction
Y_predict_p = GRNNet.predict(X_p) #for any prediction: GRNNet.predict([[0.5,0.1,0.2]])
Y_ts=((Y_p.values * minmax_p.emf + minval_p.emf).round(3))
Y_ps=((Y_predict_p * minmax_p.emf + minval_p.emf)[:,0].round(3))
print("Comparison Whole Dataset:")
print("Test: {0}".format((Y_p.values * minmax_p.emf + minval_p.emf).round(3)))
print("")
print("Predict: {0}".format((Y_predict_p * minmax_p.emf + minval_p.emf)[:,0].round(3)))
print("")
print("Accuracy: {0}".format(metrics.r2_score(Y_p, Y_predict_p).round(2)))
print("")