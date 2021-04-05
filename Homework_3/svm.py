#-------------------------------------------------------------------------
# AUTHOR: David Lao
# FILENAME: svm.py
# SPECIFICATION: Program uses SVM to make predictions and the highest accuracy is found as well as the combination of parameters.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1:])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
final = ''
for v_c in c: #iterates over c
    for dg in degree: #iterates over degree
        for kl in kernel: #iterates kernel
           for dfs in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=v_c, degree=dg, kernel=kl, decision_function_shape=dfs)

                #Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                accuracy = 0
                count = 0
                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                for testSample in dbTest:
                    class_predicted = clf.predict([testSample[:-1]])
                    class_predicted = int(class_predicted[0])

                    if class_predicted == int(testSample[-1]):
                        count += 1
                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here

                accuracy = count / len(dbTest)
                if accuracy > highestAccuracy:
                    highestAccuracy = accuracy
                    print(f"Highest SVM accuracy so far: {highestAccuracy}, Parameters: a={v_c}, degree={dg}, kernel= {kl}, decision_function_shape = {dfs}")
                    final = (f"Highest SVM accuracy: {highestAccuracy}, Parameters: a={v_c}, degree={dg}, kernel= {kl}, decision_function_shape = {dfs}")

#print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
#--> add your Python code here

print(final)












