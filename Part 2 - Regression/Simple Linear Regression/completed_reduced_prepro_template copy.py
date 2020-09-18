#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# matrix of features (independant variables)
X=dataset.iloc[:, :-1].values

# Create the dependant variable vector
Y=dataset.iloc[:,3].values



#Splitting the data into training and test sets
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)