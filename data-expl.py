# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:19:24 2021

@author: hodam
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

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
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state = 1)

#Fit model
melbourne_model.fit(X, y)
print("Making predictions for the following 5 houses: ")
print(X.head())
print("The predictions are: ")
print(melbourne_model.predict(X.head()))