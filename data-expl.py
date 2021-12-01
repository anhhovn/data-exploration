# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:19:24 2021

@author: hodam
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#import file path
melbourne_file_path = '../data-exploration/melb_data.csv'
#read file to a variable for easier access
melbourne_data = pd.read_csv(melbourne_file_path)
#drop value-missing row
melbourne_data.dropna(axis=0)
#describe table and export the result to csv 
#print(melbourne_data.describe())
melb_des = melbourne_data.describe().to_csv('../data-exploration/melb_des.csv')
#prediction target
y = melbourne_data.Price
#features
melbournes_features = ['Rooms', 'Bathroom','Landsize','Lattitude', 'Longtitude']
X = melbourne_data[melbournes_features]

#split data into training and validation data, for both features and targets
#the random_state argument guarantees we get the same split every time we run
#this script

train_X, val_X, train_Y, val_Y = train_test_split(X,y,random_state=0)

#Define model
melbourne_model = DecisionTreeRegressor()

#Fit model
melbourne_model.fit(train_X,train_Y)
# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_Y, val_predictions))