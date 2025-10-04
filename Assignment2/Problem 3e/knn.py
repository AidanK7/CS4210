#-------------------------------------------------------------------------
# AUTHOR: Aidan Kumar
# FILENAME: knn.py
# SPECIFICATION: LOO-CV calculation using 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: 40 Minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())
errors = 0

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    testSample = []

    for instance in db:
        if instance is not i:
            features = [float(f) for f in instance[:20]] #skips the classifier entry
            X.append(features)
        elif instance is i:
            testSample = [float(f) for f in instance[:20]]

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    true_label = 0

    for instance in db:
        if instance is not i:
            if instance[20] == "ham":
                Y.append(0)
            elif instance[20] == "spam":
                Y.append(1)
        elif instance is i:
            if instance[20] == "ham":
                true_label = 0
            elif instance[20] == "spam":
                true_label = 1

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    # ADDED above -> for instance in db loop

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != true_label:
        errors += 1

#Print the error rate
#--> add your Python code here
error_rate = errors / len(db)
print(f"LOO-CV Error rate for 1NN: {error_rate}")




