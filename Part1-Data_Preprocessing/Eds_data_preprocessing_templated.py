#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# matrix of features (independant variables)
X=dataset.iloc[:, :-1].values

# Create the dependant variable vector
Y=dataset.iloc[:,3].values

# Replacing Missing data with the mean of that column // Hit cmd+i to see information on libraries (Have to use it in the console as it doesn't work in the editor)
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy='mean', verbose=0)
missingvalues = missingvalues.fit(X[: , 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding Y data
Y = LabelEncoder().fit_transform(Y)

#Splitting the data into training and test sets
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)