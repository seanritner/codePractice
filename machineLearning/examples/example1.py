## Copilot Generated
# Sure, here's a simple example of how you can use Python and the Scikit-learn library to experiment with basic Machine Learning principles.
#This example uses the Iris dataset, which is a multivariate dataset introduced by the British statistician and biologist Ronald Fisher in his 1936 paper.
# This code will load the dataset, split it into training and testing sets, train a Decision Tree classifier, and then make predictions on the testing set.

# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = datasets.load_iris()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1) # 70% training and 30% testing

# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:",accuracy_score(y_test, y_pred))

#This is a supervised learning example because we're using labeled data (the Iris dataset) to train our model. The labels are the species of each Iris flower.
#Remember to install the necessary libraries before running the code. You can do this with pip:

#pip install -U scikit-learn

#This is a very basic example, but it should give you a good starting point to experiment with Machine Learning in Python. Happy coding!
