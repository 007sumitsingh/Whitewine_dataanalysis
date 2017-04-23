#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn import linear_model, metrics, tree, ensemble, svm, preprocessing
import math


#function to split dataset into test and train 
def split(df, train_percent=.6, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    train = df.ix[perm[:train_end]]
    test = df.ix[perm[train_end:]]
    return train, test


data = pd.read_csv("winequality-white.csv", sep = ';')
#To get correlation table
corr = data.corr()
# To classify all wines into bad, normal and good depending upon whether their quality is less than, equal to, or greater than 6)
data['taste'] = data.quality.apply(lambda x: {0:"bad", 1:"bad", 2:'bad', 3:"bad", 4:"bad", 5:"bad", 6:"normal", 7:"good", 8:"good", 9:"good", 10:"good"}[x])


#Separating Dataset into training and testing datasets (60%, 40% split)
train, test = split(data)


#Building Multiple Linear Regression Model 
lm = linear_model.LinearRegression()
lm.fit(train[train.columns[:11]], train[train.columns[11]])
predicted = lm.predict(test.drop(["quality", "taste"], axis = 1))
rmse = math.sqrt(metrics.mean_squared_error(test.quality, predicted))
# Computing RMSE (Root Mean Square Error)
print("RMSE for Linear Model: ", rmse)



#Building Regression Tree Model
decision_tree_model = tree.DecisionTreeRegressor()
decision_tree_model.fit(train[train.columns[:11]], train[train.columns[11]])
predicted = decision_tree_model.predict(test.drop(["quality", "taste"], axis = 1))
rmse = math.sqrt(metrics.mean_squared_error(test.quality, predicted))
# Computing RMSE (Root Mean Square Error)
print("RMSE for Decision Tree: ", rmse)


#Building Random Forest Model
random_forest_model = ensemble.RandomForestClassifier(n_estimators = 500, n_jobs = 3)
random_forest_model.fit(train[train.columns[:11]], train[train.columns[12]])
predicted = random_forest_model.predict(test.drop(["quality", "taste"], axis = 1))
cm = metrics.confusion_matrix(test.taste, predicted)
accuracy = cm.diagonal().sum()/cm.sum()
# Precision - Fractions of correct predicitions for a class
precision = cm.diagonal()/cm.sum(axis = 0)
# Recall - Fractions of instances of a class that were correctly predicted
recall = cm.diagonal()/cm.sum(axis = 1)
f1 = 2*precision*recall/(precision+recall)
print("Accuracy of Random Forest: ", accuracy)
print(pd.DataFrame([precision, recall, f1], index = ['precision', 'recall', 'f1'], columns = ['bad', 'good', 'normal']).T)



#Building SVM Classification Model 
SVM = svm.SVC()
SVM.fit(train[train.columns[:11]], train[train.columns[12]])
predicted = SVM.predict(test.drop(["quality", "taste"], axis = 1))
cm = metrics.confusion_matrix(test.taste, predicted)
accuracy = cm.diagonal().sum()/cm.sum()
# Precision - Fractions of correct predicitions for a class
precision = cm.diagonal()/cm.sum(axis = 0)
# Recall - Fractions of instances of a class that were correctly predicted
recall = cm.diagonal()/cm.sum(axis = 1)
f1 = 2*precision*recall/(precision+recall)
print("Accuracy of SVM: ", accuracy)
print(pd.DataFrame([precision, recall, f1], index = ['precision', 'recall', 'f1'], columns = ['bad', 'good', 'normal']).T)




#Applying Models with Normalization and Outlier Removal 
# (No Outliers present in total sulphar dioxide and alcohol features)
quality = data.pop('quality')
taste = data.pop('taste')
data = data.apply(lambda col: preprocessing.scale(col))
data['quality'] = quality
data['taste'] = taste
data = data[data[data.columns[:11]].apply(max, axis = 1)<3]
data = data[data[data.columns[:11]].apply(min, axis = 1)>-3]
train, test = split(data)



#Building Multiple Linear Regression Model 
lm = linear_model.LinearRegression()
lm.fit(train[train.columns[:11]], train[train.columns[11]])
predicted = lm.predict(test.drop(["quality", "taste"], axis = 1))
rmse = math.sqrt(metrics.mean_squared_error(test.quality, predicted))
# Computing RMSE (Root Mean Square Error)
print("RMSE for Linear Model: ", rmse)



#Building Decision Tree Model
decision_tree_model = tree.DecisionTreeRegressor()
decision_tree_model.fit(train[train.columns[:11]], train[train.columns[11]])
predicted = decision_tree_model.predict(test.drop(["quality", "taste"], axis = 1))
rmse = math.sqrt(metrics.mean_squared_error(test.quality, predicted))
# Computing RMSE (Root Mean Square Error)
print("RMSE for Decision Tree: ", rmse)




#Building Random Forest Model

random_forest_model = ensemble.RandomForestClassifier(n_estimators = 500, n_jobs = 3)
random_forest_model.fit(train[train.columns[:11]], train[train.columns[12]])
predicted = random_forest_model.predict(test.drop(["quality", "taste"], axis = 1))
cm = metrics.confusion_matrix(test.taste, predicted)
accuracy = cm.diagonal().sum()/cm.sum()
# Precision - Fractions of correct predicitions for a class
precision = cm.diagonal()/cm.sum(axis = 0)
# Recall - Fractions of instances of a class that were correctly predicted
recall = cm.diagonal()/cm.sum(axis = 1)
f1 = 2*precision*recall/(precision+recall)
print("Accuracy of Random Forest: ", accuracy)
print(pd.DataFrame([precision, recall, f1], index = ['precision', 'recall', 'f1'], columns = ['bad', 'good', 'normal']).T)



#Building SVM Classification Model 

SVM = svm.SVC()
SVM.fit(train[train.columns[:11]], train[train.columns[12]])
predicted = SVM.predict(test.drop(["quality", "taste"], axis = 1))
cm = metrics.confusion_matrix(test.taste, predicted)
accuracy = cm.diagonal().sum()/cm.sum()
# Precision - Fractions of correct predicitions for a class
precision = cm.diagonal()/cm.sum(axis = 0)
# Recall - Fractions of instances of a class that were correctly predicted
recall = cm.diagonal()/cm.sum(axis = 1)
f1 = 2*precision*recall/(precision+recall)
print("Accuracy of SVM: ", accuracy)
print(pd.DataFrame([precision, recall, f1], index = ['precision', 'recall', 'f1'], columns = ['bad', 'good', 'normal']).T)
