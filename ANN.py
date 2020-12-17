###########################################################################################################
######################### PART-1  DO THE PRE-REQUISITES  ##################################################
###########################################################################################################
# TYPE THESE IN THE IPYTHON CONSOLE OF THE SPYDER NOTEBOOK UNDER ANACONDA NAVIGATOR
###########################################################################################################

# 0. Install git            :-      conda install git

# 1. Install THEANO         :-      pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# 2. Install TENSORFLOW     :-      pip install tensorflow

# 3. Install KERAS          :-      pip install --upgrade keras

###########################################################################################################
######################### PART-2 DATA PREPROCESSING  ######################################################
###########################################################################################################

############################# I> IMPORT THE LIBRARIES  ####################################################

import numpy as np # pip install numpy==1.19.3- Try using this in case of any failure in sanity checks as Python 3.7 is incompatible
import matplotlib.pyplot as plt
import pandas as pd

############################# II> IMPORT THE DATASET  #####################################################

dataset = pd.read_csv('E:\\DESK PROJECTS\\Github\\DATASETS\\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

############################# III> ENCODE CATEGORICAL DATA  ###############################################

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype = np.str)
X = X[:, 1:]

############################# IV> SPLIT THE DATASET INTO TRAIN AND TEST SETS  #############################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

############################# V> FEATURE SCALING  #########################################################

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###########################################################################################################
######################### PART-3 MAKING THE ANN MODEL  ####################################################
###########################################################################################################

############################# I> IMPORT THE LIBRARIES AND PACKAGES ########################################

import keras # conda install keras==2.3.1-> try using this command in case of any import errors
from keras.models import Sequential
from keras.layers import Dense

############################# II> INITIALIZE THE ANN   ####################################################

classifier = Sequential()

############################# III> ADD INPUT LAYER AND 1ST HIDDEN LAYER ###################################

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

##############################  IV> ADD THE SECOND HIDDEN LAYER ###########################################

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

############################# V> ADD THE OUTPUT LAYER  ####################################################

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

############################ VI> COMPILE THE ANN  #########################################################

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

########################### VII> FIT ANN TO TRAINING SET  #################################################

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100) # This will take a while..

###########################################################################################################
######################### PART-4 MAKE PREDICTIONS   #######################################################
###########################################################################################################

########################### I> PREDICT THE TEST SET RESULTS  ##############################################

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred

########################### II> PREDICT A SINGLE NEW OBSERVATION  #########################################

"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 50, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
new_prediction

########################### III> CONFUSION MATRIX  ########################################################

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)
recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)