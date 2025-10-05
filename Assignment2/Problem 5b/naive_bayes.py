#-------------------------------------------------------------------------
# AUTHOR: Aidan Kumar
# FILENAME: naive_bayes.py
# SPECIFICATION: naive bayes calculator
# FOR: CS 4210- Assignment #2
# TIME SPENT: 25 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
Y = []

outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain':3}
temperature_map = {'Hot': 1, 'Mild': 2, 'Cool':3}
humidity_map = {'High': 1, 'Normal': 2}
wind_map = {'Strong': 1, 'Weak': 2}
playTennis_map = {'Yes': 1, 'No': 2}

for row in dbTraining:
    features = [outlook_map[row[1]], temperature_map[row[2]], humidity_map[row[3]], wind_map[row[4]]]
    X.append(features)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
for row in dbTraining:
    Y.append(playTennis_map[row[5]])

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB(var_smoothing=0.000000001)
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for test_row in dbTest:
    test_attributes = [outlook_map[test_row[1]], temperature_map[test_row[2]], humidity_map[test_row[3]], wind_map[test_row[4]]]

    probability = clf.predict_proba([test_attributes])[0]
    prediction = clf.predict([test_attributes])[0]

    print(probability, prediction)

