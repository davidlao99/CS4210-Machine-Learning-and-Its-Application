#-------------------------------------------------------------------------
# AUTHOR: David Lao
# FILENAME: naive_bayes.py
# SPECIFICATION: using the naive bayes algorithm to predict the class and find the accuracy
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
import csv

from sklearn.naive_bayes import GaussianNB

feature = {
    'Sunny': 1,
    'Overcast': 2,
    'Rain': 3,
    'Hot': 4,
    'Mild': 5,
    'Cool': 6,
    'High': 7,
    'Normal': 8,
    'Weak': 9,
    'Strong': 10
}

Class = {
    'Yes': 1,
    'No': 2
}

dbTraining = []
X = []
Y = []

#reading the training data
#--> add your Python code here
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTraining.append (row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
X = list(map(lambda x : list(map(lambda y : feature[y],x[1:-1])), dbTraining))
Y = list(map(lambda x : Class[x[-1]], dbTraining))
# for r in dbTraining:
#     row = r[1:-1]
#     X.append([feature[row[0]], feature[row[1]], feature[row[2]], feature[row[3]]])
#     Y.append(Class[r[-1]])

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here

dbTest = []

with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTest.append(row)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]

for r in dbTest:
    row = r[1:-1]
    predicted = clf.predict_proba([[feature[row[0]], feature[row[1]], feature[row[2]], feature[row[3]]]])[0]
    accuracy = 0
    pred_class = ''
    if predicted[0] > predicted[1]:
        pred_class = "Yes"
        accuracy = predicted[0]
    else:
        pred_class = "No"
        accuracy = predicted[1]

    if accuracy < 0.75: continue
    print((r[0]).ljust(15) + (r[1]).ljust(15) + (r[2]).ljust(15) + (r[3]).ljust(15) + (r[4]).ljust(15) + (pred_class).ljust(15) + str(accuracy).ljust(15))


    # print('Predicted Solution for D'': ',predicted)


