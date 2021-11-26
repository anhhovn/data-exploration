# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:19:24 2021

@author: hodam
"""

import pandas as pd
#import file path
melbourne_file_path = '../data-exploration/melb_data.csv'
#read file to a variable for easier access
melbourne_data = pd.read_csv(melbourne_file_path)
#drop value-missing row
melbourne_data.dropna(axis=0)
#describe table and export the result to csv 
#print(melbourne_data.describe())
melbourne_data.describe().to_csv('../data-exploration/melb_des.csv')