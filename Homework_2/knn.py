#-------------------------------------------------------------------------
# AUTHOR: David Lao
# FILENAME: knn.py
# SPECIFICATION: using the knn algorithm to predict the test case class and find the error rate
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

Y_dict = {
    '+': 1,
    '-': 2
}
err_count = 0.0
#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]
    #--> add your Python code here
    X = []
    Y = []
    for row in db:
        if row == instance: continue
        X.append(row[:-1])
        Y.append(Y_dict[row[-1]])

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]
    #--> add your Python code here
    # Y =

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = instance

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample[:-1]])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != Y_dict[testSample[-1]]:
        err_count += 1.0

#print the error rate
#--> add your Python code here
error_rate = err_count / len(db)
print('Error Rate:', error_rate)






