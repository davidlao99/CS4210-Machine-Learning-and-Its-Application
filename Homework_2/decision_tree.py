#-------------------------------------------------------------------------
# AUTHOR: David Lao
# FILENAME: decision_tree.py
# SPECIFICATION: create a decision tree for each data set and find the accuracy of each decision tree with a test set
# FOR: CS 4210- Assignment #2
# TIME SPENT: 45 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

X_dict = {
    'Young': 1,
    'Prepresbyopic': 2,
    'Presbyopic': 3,
    'Myope': 4,
    'Hypermetrope': 5,
    'No': 6,
    'Yes': 7,
    'Reduced': 8,
    'Normal': 9
}

Y_dict = {
    'Yes': 1,
    'No': 2
}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    for row in dbTraining:
        X.append([X_dict[row[0]], X_dict[row[1]], X_dict[row[2]], X_dict[row[3]]])
        Y.append(Y_dict[row[-1]])

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =

    low_acc = 1.0
    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest = []
       acc_count = 0.0

       with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for i, row in enumerate(reader):
               if i > 0:  # skipping the header
                   dbTest.append(row)

       for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            class_predicted = clf.predict([[X_dict[data[0]], X_dict[data[1]], X_dict[data[2]], X_dict[data[3]]]])[0]

           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            if class_predicted == Y_dict[data[-1]]:
                acc_count += 1.0

        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
       accuracy = float(acc_count / len(dbTest))
       if accuracy < low_acc:
            low_acc = accuracy


    #print the lowest accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that:
         #final accuracy when training on contact_lens_training_1.csv: 0.2
         #final accuracy when training on contact_lens_training_2.csv: 0.3
         #final accuracy when training on contact_lens_training_3.csv: 0.4
    #--> add your Python code here

    print(ds,': ', low_acc)




