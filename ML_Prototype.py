"""
In The Name of GOD



@author: Ali Pilehvar Meibody




Description :
This is prototype of a machine learning model that will be used to predict 
and you can change some of the code and use it in your own project.


"""


#==============================================================================
'''
                       Importing the necessary libraries
'''
#============================================================================== 
#---Numpy , matplotlib , Pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Regressor Models
from sklearn.linear_model import LinearRegression
from sklarn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

#classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.metrics import accuracy_score


#data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer



#==============================================================================
'''
                       Loading the data
'''
#==============================================================================

#******
#here based on your data path you can change the path
path = '....'

#if it is .csv file -->
data = pd.read_csv(path)

#if it is .xlsx or .xls file -->
data = pd.read_excel(path)


#THEN You can check columns name
data.columns


data.head()


data.tail()


#you can check the data types of the columns
data.info()


'''
1-Empty cells
you can use data.dropna() , data.fillna() , data.interpolate() , data.ffill() , data.bfill()

2-Wrong data types
you can use data.astype() to change the data types of the columns

3-Wrong data values
you can check with data.describe() to see the summary of the data

4-Duplicate data
you can use data.drop_duplicates() to remove the duplicate data

5-Outliers
you can use data.boxplot() to see the outliers

6-Correlation
you can use data.corr() to see the correlation between the columns


7-Data imbalance
you can use data.value_counts() to see the value counts of the columns

8-Data distribution
you can use data.hist() to see the distribution of the data



'''


#optionally you can save the cleaned data to a new file
data.to_csv('cleaned_data.csv', index=False)
#or
data.to_excel('cleaned_data.xlsx', index=False)



#==============================================================================
'''
                       First you can conver that to x and y variables
'''
#==============================================================================
'''
Here you have only  data which is dataframe
you must know we have X which is independent variables and is our process parameters and ...

also you have y which is dependent variable and is our response variable and quality parameters


for faster and better procesing you can conver to numpy arrays


'''
#******
#here you can change the columns name based on your data
x_columns = ['column1', 'column2', 'column3', 'column4', 'column5']
y_column = 'column6'


x_data = data[x_columns]
y_data = data[y_column]


x = np.array(x_data)

#if your x has only one column you can use this code
#x = np.array(x_data).reshape(-1, 1)


y = np.array(y_data)




#==============================================================================
'''
                       Train Test split
'''
#==============================================================================
#we import that at the beginning of the code
#from sklearn.model_selection import train_test_split

#you can also change the test_size and random_state
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



#==============================================================================
'''
                       Scaling the data
'''
#==============================================================================
#we import that at the beginning of the code
#from sklearn.preprocessing import StandardScaler

#you can also ignore this step if you want 
#you can use one of the following methods to scale the data
#1-MinMaxScaler
#2-MaxAbsScaler
#3-RobustScaler
#4-Normalizer

scaler = StandardScaler()
scaler = MinMaxScaler()
scaler = RobustScaler()
scaler = Normalizer()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#so you x is now scaled (also if you ignore , it is ok)


#==============================================================================
'''
                       Choosing The model
'''
#==============================================================================

'''
For regression problems you can use the following models:
1-LinearRegression
2-KNeighborsRegressor
3-DecisionTreeRegressor
4-RandomForestRegressor
5-SVR
6-MLPRegressor
'''

#we imported all of them at the beginning of the code
#from sklearn.linear_model import LinearRegression
#from sklarn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.svm import SVR
#from sklearn.neural_network import MLPRegressor

#so you can choose the model based on your data and your problem
model = LinearRegression()
model = KNeighborsRegressor()
model = DecisionTreeRegressor()
model = RandomForestRegressor()
model = SVR()
model = MLPRegressor()





'''
For classification problems you can use the following models:   

1-LogisticRegression
2-KNeighborsClassifier
3-DecisionTreeClassifier
4-RandomForestClassifier
5-SVC
6-MLPClassifier
'''
#we imported all of them at the beginning of the code
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier

#so you can choose the model based on your data and your problem
model = LogisticRegression()
model = KNeighborsClassifier()
model = DecisionTreeClassifier()
model = RandomForestClassifier()
model = SVC()
model = MLPClassifier()


'''
***
for More Notes on hyperparamters you can check at the end of code 
'''



#==============================================================================
'''
                       Training the model
'''
#==============================================================================
#after that we can 

#firstly you train your model with x_train and y_train
model.fit(x_train, y_train)

#then you can test your model with x_test and y_test
y_pred = model.predict(x_test)


#based on MAE or MAPE you can check the accuracy of your model
#also we import that at the beginning of the code
#from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Training MAE: {mae}')
print(f'Training MAPE: {mape}')

'''

This trianing scores is show how much your model is learning from your data
and it is not related to prediction , it  is only for training the model
if it is low --> underfitting --> means that it doesn't learn from your data well
so it has test-score low too .

but if it is too high also is not good , it means that it is overfitting and it doesn't generalize well

so you must trade off between underfitting and overfitting
'''


#==============================================================================
'''
                       Testing the model (validation)
'''
#==============================================================================

#firstly you test your model with x_test and y_test
model.fit(x_test, y_test)
y_pred = model.predict(x_test)

#based on MAE or MAPE you can check the accuracy of your model
#also we import that at the beginning of the code
#from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Testing MAE: {mae}')
print(f'Testing MAPE: {mape}')






#==============================================================================
'''
                       Usage phase
'''
#==============================================================================

'''
Now you can use your model to predict new data
and also you can graph or anything that you wnat
'''
new_data = np.array([[1, 2, 3, 4, 5]])
#if you have scaling data , if not ignore this line
new_data = scaler.transform(new_data)



new_pred = model.predict(new_data)

print(f'New prediction: {new_pred}')


#==============================================================================
'''
                       Saving the model
'''
#==============================================================================
#you can save your model to a file
model.save('model.pkl')
#or
model.save('model.joblib')




'''

Appendix A : More on hyperparamters


at the step of definition of the model we must set the hyperparamters

for example :

model = KNeighborsRegressor(n_neighbors=5, weights='uniform')
model = DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1)
model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1)
model = SVR(C=1.0, kernel='rbf', gamma='scale')
model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam')



but what is theese hyperparametrs?
actually it doesnt matter much , you can set them to any value you want
we must evaluate and trade off which is in detailed discussed in Advanced_Ml_protoype.py file



1- Linear regression : No hyperparamters

2- KNN : n_neighbors , weights 

n_neighbors : number of neighbors to consider, effect : if it is low --> underfitting , if it is high --> overfitting
weights : weights of the neighbors, effect : if it is 'uniform' --> all neighbors have the same weight, if it is 'distance' --> closer neighbors have more weight

3- Decision Tree : max_depth , min_samples_split , min_samples_leaf

max_depth : maximum depth of the tree, effect : if it is low --> underfitting , if it is high --> overfitting
min_samples_split : minimum number of samples required to split an internal node, effect : if it is low --> underfitting , if it is high --> overfitting
min_samples_leaf : minimum number of samples required to be a leaf node, effect : if it is low --> underfitting , if it is high --> overfitting

4- Random Forest : n_estimators , max_depth , min_samples_split , min_samples_leaf

n_estimators : number of trees in the forest, effect : if it is low --> underfitting , if it is high --> overfitting
max_depth : maximum depth of the tree, effect : if it is low --> underfitting , if it is high --> overfitting
min_samples_split : minimum number of samples required to split an internal node, effect : if it is low --> underfitting , if it is high --> overfitting
min_samples_leaf : minimum number of samples required to be a leaf node, effect : if it is low --> underfitting , if it is high --> overfitting

5- SVM : C , kernel , gamma

C : regularization parameter, effect : if it is low --> underfitting , if it is high --> overfitting
kernel : kernel type, effect : if it is 'linear' --> linear kernel, if it is 'poly' --> polynomial kernel, if it is 'rbf' --> radial basis function kernel
gamma : kernel coefficient, effect : if it is low --> underfitting , if it is high --> overfitting


6- MLP : hidden_layer_sizes , activation , solver

hidden_layer_sizes : number of neurons in the hidden layers, effect : if it is low --> underfitting , if it is high --> overfitting
activation : activation function, effect : if it is 'relu' --> ReLU, if it is 'sigmoid' --> sigmoid, if it is 'tanh' --> tanh
solver : solver for optimization, effect : if it is 'adam' --> Adam, if it is 'sgd' --> SGD, if it is 'lbfgs' --> LBFGS

'''