# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#use to convert variables(string) into values using scikitlearn(labelencoder)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#some of the variables are coded as 0 1 and 2 but that variables are not define with ranking.
#so dummy variable are introduce inplace of that
#oneHotEncoder for dummy variable creation using scikitlearn
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#always testing and training required. training to train the data and test to validate it.

#will use train_test_split to divide our data. training set split ratio taken as 80:20.
#standards are 80:20,75:25,60:40
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#scaling of data is required as some of the data is very large and some of the data is very small.
#standardScalar is use to scale the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#ready with data set to apply nn.



# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#naming it as classifier. using sequencial module for initialization

#Initializing Neural Network
classifier = Sequential()

#using dense function will add hidden layer one by one otherwise it will take bit effort


#output_dim:-no of nodes we want to add to this layer
#init:- initialization of stochastic gradient decent
#in neural network want to assign weight to each each mode which is nothing but the importance of that node.
#initialially we are randomly initializing weights using uniform function.
#input_dim:- only use for first layer as model doesnt know the no of out input variables.
#in our case total input are 11. in 2nd layer system automatically knows about the input from the first hidden layer.
#activation function:-neuron applies activation function to weighted sum(summation of Wi*Xi where w is weight and x is input and i is suffix of w and x).
#if any neuron have value nearly 1 then it will be more activated and more passes signal.
#CTRL_BREAK_EVENT which activation function use is crucial task.
#using rectifier(relu) function in hidden layer and sigmoid function in output layer as we require binary result from output layer.
#if no of category in output layer is more than 2 then use softmax function.

# Adding the input layer and the first hidden layer
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#optimizer:-use to find optimal set of weights. algo is stochastic gradient descent(SGD).
#we will use adam algo in SGD.
#sgd depnds on loss thus our second parameter is loss.
#our dependent variable is binary so we have to use binary_crossentropy.
#for more than 2 category use categorical_crossentropy.
#to improve performance of neural network add metrics as accuracy


# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#deep learning neaural network is formed

#batch size is used to specify the no of observations after which you want to update weight.
#epoch is total no of iterations.
#choosing value of batch size and epoch is trial and error there is no specific rule for that.
"""
# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#predicting the test result.
#prediction function gives us the probability of the customer leaving the company.

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#evaluating our model performance.
#to check the accuracy build confusion matrix.

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""







#finding the accuracy of model.

