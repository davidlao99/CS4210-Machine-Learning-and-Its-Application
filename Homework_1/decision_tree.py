#-------------------------------------------------------------------------
# AUTHOR: David Lao
# FILENAME: decision_tree.py
# SPECIFICATION: Creates decision tree using sklearn with data from csv represented as numerical values
# FOR: CS 4200- Assignment #1
# TIME SPENT: 1hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays
#importing some Python libraries

from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)
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
#transfor the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here

#transfor the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =

for row in db:
    X.append([X_dict[row[0]], X_dict[row[1]], X_dict[row[2]], X_dict[row[3]]])
    Y.append(Y_dict[row[len(row)-1]])


#fiiting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()


