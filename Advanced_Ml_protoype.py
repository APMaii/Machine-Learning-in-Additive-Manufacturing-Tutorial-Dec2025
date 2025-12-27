"""
In The Name of GOD



@author: Ali Pilehvar Meibody




Description :
First look at the ML_Prototype.py file and then you can understand this file.
This file is more advanced and includes more features and more models.


what is different here ?

actually the only difference is

1-instead of using training and test data , we use cross-validation to evaluate the model (what is that? then we talk)

2-instead of using only one model and one hyperparamter , we use GridSearchCV to find the best model and the best hyperparamter




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


#cross-validation
from sklearn.model_selection import GridSearchCV


#==============================================================================
'''
                       Loading the data
'''
#==============================================================================
###### everything is same with ML_Prototype.py file

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
###### everything is same with ML_Prototype.py file

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
###### everything is same with ML_Prototype.py file

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#we don't have train test split hre, we use cross validation instead
'''
what is cross validation?
cross validation is a technique to evaluate the performance of a model
it is a technique to evaluate the performance of a model by splitting the data into k folds
and then training the model on k-1 folds and testing on the remaining fold
and then repeat the process k times and then average the results

so we don't have train test split here, we use cross validation instead

more details --> for instance we have 10 data (1,2,3,4,5,6,7,8,9,10) and we want to use 8 for training and 2 for testing

in fold1 --> 1,2,3,4,5,6,7,8 --> training data , 9,10 --> testing data
in fold2 --> 1,2,3,4,5,6,9,10 --> training data , 7,8 --> testing data
in fold3 --> 1,2,3,4,7,8,9,10 --> training data , 5,6 --> testing data
in fold4 --> 1,2,5,6,7,8,9,10 --> training data , 3,4 --> testing data
in fold5 --> 3,4,5,6,7,8,9,10 --> training data , 1,2--> testing data

so we have 5 folds and we train the model on 8 data and test on 2 data
and then we repeat the process 5 times and then average the results

so we have 5 scores for each model and we can average them to get the final score


what is benefits of cross validation?
1-we use all of the data for training and testing
2-we get a more accurate score for the model
3-we can use the model to predict new data
4-we can use the model to tune the hyperparamters
5-we can use the model to compare different models
6-we can use the model to compare different hyperparamters
7-we can use the model to compare different data
8-we can use the model to compare different models

we use them as cv inside one thing named grid search cv

'''


#==============================================================================
'''
                       Scaling the data
'''
#==============================================================================
#here we can scale all data at once
scaler = StandardScaler()
scaler = MinMaxScaler()
scaler = RobustScaler()
scaler = Normalizer()

x = scaler.fit_transform(x)


#==============================================================================
'''
                       Choosing The model
'''
#==============================================================================
#it is same with ML_Prototype.py file

#so you can choose the model based on your data and your problem
model = LinearRegression()
model = KNeighborsRegressor()
model = DecisionTreeRegressor()
model = RandomForestRegressor()
model = SVR()
model = MLPRegressor()


#or for classification problems you can use the following models:
#so you can choose the model based on your data and your problem
model = LogisticRegression()
model = KNeighborsClassifier()
model = DecisionTreeClassifier()
model = RandomForestClassifier()
model = SVC()
model = MLPClassifier()


#==============================================================================
'''
                       Training the model
'''
#==============================================================================
#after that we can 
#here instead of using model and train them 
#from sklearn.model_selection import GridSearchCV
# we can use it to find the best model and the best hyperparamter


#for example for KNN Model 
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
}
#we wrap the model inside gridsearch and also cv
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

'''

model --> is that model that we want to train
cv --> it is the number of folds that we want to use for cross validation
scoring --> it is the metric that we want to use to evaluate the model [neg_mean_squared_error , mean_absolute_error , mean_absolute_percentage_error , r2_score]
param_grid --> it is the hyperparamters that we want to tune
'''




#instead of model fittin on train and then test we can do 

grid_search.fit(x, y)



#==============================================================================
'''
                       Evaluating the model
'''
#==============================================================================

#you can see the results of the grid search
results = grid_search.cv_results_




#for knowing what is the bets parameters 
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')


#you want to know the score that achived by the best parameters
best_score = grid_search.best_score_
print(f'Best score: {best_score}')



#==============================================================================
'''
                       Usage phase
'''
#==============================================================================


#gs is like model --> it has .predict() or anything like that

new_x = np.array([[1, 2, 3, 4, 5]])
new_x = scaler.transform(new_x)
new_pred = grid_search.predict(new_x)


#==============================================================================
'''
                       Saving the model
'''
#==============================================================================
#you can save your model to a file
grid_search.best_estimator_.save('model.pkl')




'''

Appendix A : More on param_grid


in GridSearchCV we can use param_grid to tune the hyperparamters

Here for each of these models i provdie some parameters for see that


1-Linear Regression : No params

param_grid={}


2-KNN : n_neighbors , weights 

#you can change the numbers of neighbors and weights as you want
param_grid={
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
}


3-Decision Tree : max_depth , min_samples_split , min_samples_leaf

param_grid={
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

#you can change the max depth and the number of samples to split and to be a leaf as you want


4-Random Forest : n_estimators , max_depth , min_samples_split , min_samples_leaf

param_grid={
    'n_estimators': [20,40,60,80,100],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}



5-SVM : C , kernel , gamma
param_grid={
    'C': [0.1, 0.5, 1.0, 5.0, 10.0],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto'],
}

#you can change the C and the kernel and the gamma as you want

6-MLP : hidden_layer_sizes , activation , solver

param_grid={
    'hidden_layer_sizes': [(100,), (200,), (300,), (400,), (500,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
}

'''

