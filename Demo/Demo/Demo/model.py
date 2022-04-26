
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:13:26 2022

@author: Jayesh
"""

from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# =============================================================================
# Logistic Regression - Yashasviben Patel - 301207828
# =============================================================================


def LogisticRegressionModel():
    
    data = pd.read_csv(r"Bicycle_Thefts.csv")

    drop_column = ['OBJECTID', 'X', 'Y', 'event_unique_id', 'City',
               'Location_Type', 'NeighbourhoodName', 'Latitude', 'Longitude', 'OBJECTID_1']

    X1 = data['Bike_Model']
    Y1 = data['Status']

    data.drop(drop_column, axis=1, inplace=True)

    data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True)

    data['Occurrence_Date'] = pd.to_datetime(
    data['Occurrence_Date']).dt.time

    data['Bike_Colour'].fillna('other', inplace=True)

    data['Cost_of_Bike'].replace(0, np.nan, inplace=True)


    unknown_make = ['UK', 'NULL', 'UNKNOWN MAKE', 'UNKNOWN', 'NONE', 'NO', 'UNKNOWNN',
                'UNKONWN', 'UNKOWN', 'UNKNONW', '-', 'UNKNOW', 'NO NAME', '?']  # all typos stand for known
    giant = data['Bike_Make'][data['Bike_Make'].str.contains(
  'giant', case=False, na=False)].unique().tolist()  # alias of giant
    giant.append('GI')


    data['Bike_Make'].replace(giant, 'GIANT', inplace=True)
    data['Bike_Make'].replace('OT', 'OTHER', inplace=True)
    data['Bike_Make'].replace(unknown_make, np.nan, inplace=True)


    encoder = preprocessing.LabelEncoder()
    data['Bike_Type'] = encoder.fit_transform(
    data['Bike_Type'])  # only numerical values for KNNImputer
    data['Bike_Make'] = pd.Series(encoder.fit_transform(data['Bike_Make'][data['Bike_Make'].notna(
    )]), index=data['Bike_Make'][data['Bike_Make'].notna()].index)  # only numerical values for KNNImputer
    data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']])
    data[['Bike_Type', 'Bike_Speed', 'Bike_Make']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Bike_Make']])


    low = data['Cost_of_Bike'].quantile(.25)
    average = data['Cost_of_Bike'].quantile(.5)
    high = data['Cost_of_Bike'].quantile(.75)
    data['cost_catag'] = np.where(data['Cost_of_Bike'] <= low, 'low', np.where((data['Cost_of_Bike'] > low) & (
    data['Cost_of_Bike'] <= average), 'average', np.where((data['Cost_of_Bike'] > average) & (data['Cost_of_Bike'] <= high), 'high', 'luxury')))



    data['Status'].replace('STOLEN', 0, inplace=True)
    data['Status'].replace(['UNKNOWN', 'RECOVERED'], 1, inplace=True)

    print(data.head())


    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    for col in categorical_cols:
     data[col] = encoder.fit_transform(data[col])
    X, Y = data.drop('Status', axis=1), data['Status']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

#from sklearn.linear_model import LogisticRegression

    log_main = LogisticRegression()
    log_main.fit(x_train, y_train)


    y_pred = log_main.predict(x_test)
    print('Accuracy of Logistic is:', accuracy_score(y_test, y_pred))


    log_main.get_params().keys()

    LRparam_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    # 'max_iter': list(range(100,800,100)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
    LR_search = GridSearchCV(log_main, param_grid=LRparam_grid, refit = True, verbose = 3, cv=5)

# Fit the model
    LR_search.fit(x_train , y_train)

# Print out the best parameters
    LR_search.best_params_

#Print out the score of the model
    print('The training score is : ')
    print(LR_search.score(x_train, y_train))
    print('The test score is : ')
    print(LR_search.score(x_test, y_test))


# Print out the best estimator
    LR_search.best_estimator_


# Fit the test data using the fine-tuned model
    fine_tuned_model = LR_search.best_estimator_.fit(x_train, y_train)


    y_grid_pred = fine_tuned_model.predict(x_test)


#from sklearn.metrics import confusion_matrix, accuracy_score
    print('Confusion Matrix : \n', confusion_matrix(y_test, y_grid_pred))
    print('Accuracy', accuracy_score(y_test, y_grid_pred))

    return accuracy_score(y_test,y_grid_pred)
LogisticRegressionModel()