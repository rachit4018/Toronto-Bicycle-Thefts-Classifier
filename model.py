
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
import ppscore as pps1
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from lofo import LOFOImportance, Dataset, plot_importance
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
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd, numpy as np, ppscore as pps
from datetime import datetime


data = pd.read_csv("J:\Centennial Stuff\Sem2\Supervised Learning\Project_Jayesh\Bicycle_Thefts.csv")

def knnModel():
    drop_column = ['OBJECTID', 'X', 'Y', 'event_unique_id', 'City',
               'Location_Type', 'NeighbourhoodName', 'Latitude', 'Longitude', 'OBJECTID_1']

    data = pd.read_csv("J:\Centennial Stuff\Sem2\Supervised Learning\Project_Jayesh\Bicycle_Thefts.csv")

    X1 = data['Bike_Model']

    Y1 = data['Status']

    data.drop(drop_column, axis=1, inplace=True)
# Neighbourhood is identical with Hood ID
    data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True)


    data['Occurrence_Date'] = pd.to_datetime(
    data['Occurrence_Date']).dt.time  # change data type
    data['Bike_Colour'].fillna('other', inplace=True)  # fill nan value
    data['Cost_of_Bike'].replace(0, np.nan, inplace=True)  # zero is also invalid

    unknown_make = ['UK', 'NULL', 'UNKNOWN MAKE', 'UNKNOWN', 'NONE', 'NO', 'UNKNOWNN',
                'UNKONWN', 'UNKOWN', 'UNKNONW', '-', 'UNKNOW', 'NO NAME', '?']  # all typos stand for known
    giant = data['Bike_Make'][data['Bike_Make'].str.contains(
     'giant', case=False, na=False)].unique().tolist()  # alias of giant
    giant.append('GI')

    data['Bike_Make'].replace(giant, 'GIANT', inplace=True)
    data['Bike_Make'].replace('OT', 'OTHER', inplace=True)
    data['Bike_Make'].replace(unknown_make, np.nan, inplace=True)

# transform non-numeric data
    encoder = preprocessing.LabelEncoder()
    data['Bike_Type'] = encoder.fit_transform(
    data['Bike_Type'])  # only numerical values for KNNImputer
    data['Bike_Make'] = pd.Series(encoder.fit_transform(data['Bike_Make'][data['Bike_Make'].notna(
    )]), index=data['Bike_Make'][data['Bike_Make'].notna()].index)  # only numerical values for KNNImputer
    data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']])
    data[['Bike_Type', 'Bike_Speed', 'Bike_Make']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Bike_Make']])

# Convert cost to cost catagory
    low = data['Cost_of_Bike'].quantile(.25)
    average = data['Cost_of_Bike'].quantile(.5)
    high = data['Cost_of_Bike'].quantile(.75)
    data['cost_catag'] = np.where(data['Cost_of_Bike'] <= low, 'low', np.where((data['Cost_of_Bike'] > low) & (
    data['Cost_of_Bike'] <= average), 'average', np.where((data['Cost_of_Bike'] > average) & (data['Cost_of_Bike'] <= high), 'high', 'luxury')))

# upcycling of data
    data['Status'].replace('STOLEN', 0, inplace=True)
    data['Status'].replace(['UNKNOWN', 'RECOVERED'], 1, inplace=True)

    print(data.head())

# encoding categorical features
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    for col in categorical_cols:
      data[col] = encoder.fit_transform(data[col])
    X, Y = data.drop('Status', axis=1), data['Status']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)


#print('-------------- K-Nearest Neighbour  -----------')
# Jayesh Amodkar - 301211026
#---------------------------------------------------------
##PASTE HERE
#     scaler = StandardScaler()
#     scaler.fit(x_train)
#     x_train = scaler.transform(x_train)
#     x_test = scaler.transform(x_test)

# # Model fitting with K-cross Validation and GridSearchCV
#     k_range = list(range(1, 11))
#     param_grid = dict(n_neighbors=k_range)

# # Fine Tuning- defining parameter range for GridSearch kernel
#     knn = KNeighborsClassifier()
#     grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy',
#                     return_train_score=False, verbose=1)

# # fitting the model for grid search
#     grid_search = grid.fit(x_train, y_train)

# # best parameters & estimator
# #    print("Best Params: ", grid_search.best_params_)
# #    print("Best estimators are: ", grid_search.best_estimator_)

#     accuracy = grid_search.best_score_ * 100
# #    print(
# #    "Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

# # Training the KNN Classification model on the Training Set with best param
#     classifier = KNeighborsClassifier(n_neighbors=2)
#     classifier.fit(x_train, y_train)

# # Predicting the Test set results
#     y_pred = classifier.predict(x_test)

# # predict_proba to return numpy array with two columns for a binary classification for N and P
#     y_scores = classifier.predict_proba(x_test)

# # roc curve
# #fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
# #roc_auc = auc(fpr, tpr)
# ##roc_auc = auc(fpr, tpr)
# #plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
# #plt.legend(loc='lower right')
# #plt.plot([0, 1], [0, 1], 'r--')
# #plt.xlim([0, 1])
# #plt.ylim([0, 1])
# #plt.ylabel('True Positive Rate')
# #plt.xlabel('False Positive Rate')
# #plt.title('ROC Curve of kNN')
# #plt.show()

# #    print('Classification Report(N): \n', classification_report(y_test, y_pred))
# #    print('Confusion Matrix(N): \n', confusion_matrix(y_test, y_pred))
#     print('Accuracy(N): \n', metrics.accuracy_score(y_test, y_pred))
	
# 	# Comparing the Real Values with Predicted Values
#     df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
# #    print(df)    # Comparing the Real Values with Predicted Values
#     df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
# #    print(df)    
# Transforming columns- pipeline
    x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=.2)
    data_pipe = Pipeline([('tfidf', TfidfVectorizer()),
                      ('clf', KNeighborsClassifier(n_neighbors=9)), ])
    data_pipe.fit(x_train1.values.astype('U'), y_train1.values.astype('U'))

    predictions = data_pipe.predict(x_test1.values.astype('U'))
    print('\n\nClassification Report(S): \n',
      classification_report(y_test1, predictions))
#    print('Confusion Matrix(S): \n', confusion_matrix(y_test1, predictions))
 #   print('Accuracy(S): \n', metrics.accuracy_score(y_test1, predictions))

# determining non-linear correlations
#matrix = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(
#    columns='x', index='y', values='ppscore')
#sns.heatmap(matrix, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
#plt.show()

    dataset = Dataset(df=data, target='Status', features=[col for col in data.columns if col != 'Status'])
# evaluate on knn
#knn_importance = LOFOImportance(dataset, scoring='f1', model=KNeighborsClassifier())
#knn_importance = knn_importance.get_importance()
#plot_importance(knn_importance, figsize=(14, 14))
#plt.show()
    return metrics.accuracy_score(y_test1, predictions)


knnModel()


# =============================================================================
# Support Vector Machine - Rachit Pandya - 301198260
# =============================================================================
def SVMModel():
    drop_column = ['OBJECTID', 'X', 'Y', 'event_unique_id', 'City',
               'Location_Type', 'NeighbourhoodName', 'Latitude', 'Longitude', 'OBJECTID_1']

    data = pd.read_csv("J:\Centennial Stuff\Sem2\Supervised Learning\Project_Jayesh\Bicycle_Thefts.csv")

    X1 = data['Bike_Model']

    Y1 = data['Status']

    data.drop(drop_column, axis=1, inplace=True)
# Neighbourhood is identical with Hood ID
    data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True)


    data['Occurrence_Date'] = pd.to_datetime(
    data['Occurrence_Date']).dt.time  # change data type
    data['Bike_Colour'].fillna('other', inplace=True)  # fill nan value
    data['Cost_of_Bike'].replace(0, np.nan, inplace=True)  # zero is also invalid

    unknown_make = ['UK', 'NULL', 'UNKNOWN MAKE', 'UNKNOWN', 'NONE', 'NO', 'UNKNOWNN',
                'UNKONWN', 'UNKOWN', 'UNKNONW', '-', 'UNKNOW', 'NO NAME', '?']  # all typos stand for known
    giant = data['Bike_Make'][data['Bike_Make'].str.contains(
     'giant', case=False, na=False)].unique().tolist()  # alias of giant
    giant.append('GI')

    data['Bike_Make'].replace(giant, 'GIANT', inplace=True)
    data['Bike_Make'].replace('OT', 'OTHER', inplace=True)
    data['Bike_Make'].replace(unknown_make, np.nan, inplace=True)

# transform non-numeric data
    encoder = preprocessing.LabelEncoder()
    data['Bike_Type'] = encoder.fit_transform(
    data['Bike_Type'])  # only numerical values for KNNImputer
    data['Bike_Make'] = pd.Series(encoder.fit_transform(data['Bike_Make'][data['Bike_Make'].notna(
    )]), index=data['Bike_Make'][data['Bike_Make'].notna()].index)  # only numerical values for KNNImputer
    data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']])
    data[['Bike_Type', 'Bike_Speed', 'Bike_Make']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Bike_Make']])

# Convert cost to cost catagory
    low = data['Cost_of_Bike'].quantile(.25)
    average = data['Cost_of_Bike'].quantile(.5)
    high = data['Cost_of_Bike'].quantile(.75)
    data['cost_catag'] = np.where(data['Cost_of_Bike'] <= low, 'low', np.where((data['Cost_of_Bike'] > low) & (
    data['Cost_of_Bike'] <= average), 'average', np.where((data['Cost_of_Bike'] > average) & (data['Cost_of_Bike'] <= high), 'high', 'luxury')))

# upcycling of data
    data['Status'].replace('STOLEN', 0, inplace=True)
    data['Status'].replace(['UNKNOWN', 'RECOVERED'], 1, inplace=True)

    print(data.head())

# encoding categorical features
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    for col in categorical_cols:
      data[col] = encoder.fit_transform(data[col])
    X, Y = data.drop('Status', axis=1), data['Status']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

# SVM Model
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    scaler = StandardScaler()

# Combining the two transformers into a pipeline
    num_pipe_rachit = Pipeline([('Si', imp),
                            ('scalar', scaler)])
# transformer_rachit = ColumnTransformer([('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_cols)], remainder='passthrough')

    model = SVC(kernel='rbf')
    # model_rbf = SVC(kernel='rbf')
    # model_sigmoid = SVC(kernel='sigmoid')
    ###########################pipeline######################
    pipe_svm_rachit = Pipeline([
        ("scaler", num_pipe_rachit),
        ("svc", model)])
    # pipe_svm1_rachit  = Pipeline([
    #        ("transformer", transformer_rachit),
    #        ("svc", SVC(kernel='rbf'))])
    # pipe_svm2_rachit  = Pipeline([
    #          ("transformer", transformer_rachit),
    #        ("svc", SVC(kernel='sigmoid'))])

########################### train test split########################
    X_train_rachit, X_test_rachit, Y_train_rachit, Y_test_rachit = train_test_split(
    X, Y, test_size=0.2, random_state=60)
    train_pipeline = pipe_svm_rachit.fit(X_train_rachit, Y_train_rachit)
    #test_pipeline = pipe_svm_rachit.fit(X_test_rachit, Y_test_rachit)


    Y_train_predict = pipe_svm_rachit.predict(X_train_rachit)
    train_accuracy = accuracy_score(Y_train_rachit, Y_train_predict)
    print("accuracy on training data is", train_accuracy)
    f1_sc = f1_score(Y_train_rachit, Y_train_predict, average='macro')
    print(confusion_matrix(Y_train_rachit, Y_train_predict))
    print("f1 score", f1_sc)

    return accuracy_score(Y_train_rachit, Y_train_predict)

SVMModel()


########################################################################################################################
def date_diff(date1, date2):
        date_format_str = '%Y/%m/%d %H:%M:%S+00'
        start = datetime.strptime(date1, date_format_str)
        end = datetime.strptime(date2, date_format_str)
        diff = end - start
        diff_in_hours = diff.total_seconds() / 3600
        return diff_in_hours


def DCTModel():
    data = pd.read_csv("J:\Centennial Stuff\Sem2\Supervised Learning\Project_Jayesh\Bicycle_Thefts.csv")
    data.insert(len(data.columns), 'hours_to_report', 0)

    hours = []
    for row in data.itertuples(index=False):
        hours.append((date_diff(row.Occurrence_Date, row.Report_Date)
                    ) - row.Occurrence_Hour + row.Report_Hour)

    data['hours_to_report'] = hours

    #  changing months to seasons
    ['January', 'July', 'November', 'December', 'February', 'March',
    'June', 'April', 'May', 'August', 'October', 'September']

    Winter = ['December', 'January', 'February']
    Spring = ['March', 'April', 'May']
    Summer = ['June', 'July', 'August']
    Fall = ['September', 'October', 'November']

    data.loc[(data['Occurrence_Month'] .isin(Winter)), 'Season'] = 'Winter'
    data.loc[(data['Occurrence_Month'] .isin(Spring)), 'Season'] = 'Spring'
    data.loc[(data['Occurrence_Month'] .isin(Summer)), 'Season'] = 'Summer'
    data.loc[(data['Occurrence_Month'] .isin(Fall)), 'Season'] = 'Fall'


    # Get time periods from Occurence Hour


    data.loc[(data['Occurrence_Hour'] >= 0) & (
        data['Occurrence_Hour'] < 6), 'Occurrence_Time_Periods'] = 'night'
    data.loc[(data['Occurrence_Hour'] >= 6) & (
        data['Occurrence_Hour'] < 12), 'Occurrence_Time_Periods'] = 'morning'
    data.loc[(data['Occurrence_Hour'] >= 12) & (
        data['Occurrence_Hour'] < 18), 'Occurrence_Time_Periods'] = 'afternoon'
    data.loc[(data['Occurrence_Hour'] >= 18) & (
        data['Occurrence_Hour'] <= 23), 'Occurrence_Time_Periods'] = 'evening'


    # Get City from Hood Id

    data['Hood_ID'] = data['Hood_ID'].replace('NSA', 0)
    data['Hood_ID'] = pd.to_numeric(data['Hood_ID'])

    data.loc[(data['Hood_ID'] == 0), 'City'] = 'Unknown'
    data.loc[(data['Hood_ID'] > 0) & (data['Hood_ID'] <= 20), 'City'] = 'Etobicoke'
    data.loc[(data['Hood_ID'] > 20) & (
        data['Hood_ID'] <= 53), 'City'] = 'NorthYork'
    data.loc[(data['Hood_ID'] > 53) & (data['Hood_ID'] <= 61), 'City'] = 'EastYork'
    data.loc[(data['Hood_ID'] > 61) & (
        data['Hood_ID'] <= 105), 'City'] = 'OldToronto'
    data.loc[(data['Hood_ID'] > 105) & (data['Hood_ID'] <= 115), 'City'] = 'York'
    data.loc[(data['Hood_ID'] > 115) & (
        data['Hood_ID'] <= 140), 'City'] = 'Scarborough'

    # =============================================================================
    # # Bike_Colour                 2211
    # =============================================================================
    # data['Bike_Colour'] = data['Bike_Colour'].astype('string')
    data['Bike_Colour'].value_counts()  # has 256 types of different colors
    data['Bike_Colour'].fillna("OTH", inplace=True)
    color_data = data['Bike_Colour'].value_counts(dropna=False)
    major_bike_colors = ['BLK', 'BLU', 'GRY', 'WHI', 'RED', 'SIL']


    colours_data = []
    for row in data.itertuples(index=False):
        colour = ''
        if str(row.Bike_Colour).strip() in major_bike_colors:
            colour = str(row.Bike_Colour).strip()
        else:
            colour = 'OTH'
        colours_data.append(colour)

    data['Bike_Colour'] = colours_data


    # =============================================================================
    # Handling missing values
    # =============================================================================
    # Cost of bike
    data.Cost_of_Bike.value_counts(dropna=False)  # NaN        1869
    data.Cost_of_Bike.fillna(data['Cost_of_Bike'].median(
        axis=0, skipna=True), inplace=True)
    data.Cost_of_Bike.value_counts(dropna=False)


    # Bike_Model                 10221
    # gonna delete the whole column since 1/3 data is missing

    # =============================================================================
    # Drop Columns that are not useful
    # =============================================================================

    data.drop(columns=['X',
                    'Y',
                    'OBJECTID_1',
                    'OBJECTID',
                    'event_unique_id',
                    'Report_Year',
                    'Report_Date',
                    'Report_DayOfWeek',
                    'Report_DayOfMonth',
                    'Report_DayOfYear',
                    'Report_Month',
                    'Report_Hour',
                    'Bike_Model',
                    'Occurrence_Date',
                    'Occurrence_DayOfMonth',
                    'Occurrence_Year',
                    'Occurrence_Month',
                    'Occurrence_DayOfYear',
                    'Occurrence_Hour',
                    'Location_Type',
                    'Longitude',
                    'Latitude',
                    'Hood_ID',
                    'Primary_Offence',
                    'Bike_Make'
                    ], axis=1, inplace=True)

    # drop data where status = unknown
    data = data[data.Status != 'UNKNOWN']


    print(data.isnull().sum())

    # change stolen recovered to 0 and 1

    data.Status.replace(['STOLEN', 'RECOVERED'], [0, 1], inplace=True)
    # =============================================================================
    # =============================================================================
    # #  Data modeling
    # =============================================================================
    # =============================================================================


    features, target = data.drop('Status', axis=1), data.Status

    print(features.value_counts())
    print(target.value_counts())

    numerical_features = list(
        features.select_dtypes(exclude='object').columns)
    categorical_features = list(
        features.select_dtypes(include='object').columns)

    print(numerical_features)
    print(categorical_features)


    transformer_afzal = ColumnTransformer(
        [('encoder', preprocessing.OneHotEncoder(), categorical_features)], remainder='passthrough')


    clf_afzal = DecisionTreeClassifier(criterion='gini', max_depth=5)

    pipeline_afzal = Pipeline(
        [('transformer', transformer_afzal), ('decisionTree', clf_afzal)])

    x_train_afzal, x_test_afzal, y_train_afzal, y_test_afzal = train_test_split(
        features, target, test_size=0.2, random_state=60)

    pipeline = pipeline_afzal.fit(x_train_afzal, y_train_afzal)

    kf = KFold(10, shuffle=True, random_state=60)
    score = cross_val_score(pipeline_afzal, x_train_afzal, y_train_afzal, cv=kf)

    print(score)
    print(np.mean(score))

   
    parameters = {'decisionTree__min_samples_split': range(10, 300, 20), 'decisionTree__max_depth':
                range(1, 30, 2), 'decisionTree__min_samples_leaf': range(1, 15, 3)}


    clf_ran = RandomizedSearchCV(pipeline_afzal, scoring='accuracy',
                                param_distributions=parameters, cv=5, n_iter=7, refit=True, verbose=3)


    randomSearch = clf_ran.fit(x_train_afzal, y_train_afzal)

    best_estimator = randomSearch.best_estimator_

    best_estimator_clf = best_estimator.fit(x_train_afzal, y_train_afzal)

    y_pred = best_estimator_clf.predict(x_test_afzal)
    return accuracy_score(y_pred, y_test_afzal)

DCTModel()

# =============================================================================
# Neural Networks - Whitney Dsouza - 301169730
# =============================================================================
def NeuralNetworksModel():

    drop_column = ['OBJECTID', 'X', 'Y', 'event_unique_id', 'City',
               'Location_Type', 'NeighbourhoodName', 'Latitude', 'Longitude', 'OBJECTID_1']

    data = pd.read_csv("J:\Centennial Stuff\Sem2\Supervised Learning\Project_Jayesh\Bicycle_Thefts.csv")

    X1 = data['Bike_Model']

    Y1 = data['Status'] 

    data.drop(drop_column, axis=1, inplace=True)
# Neighbourhood is identical with Hood ID
    data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True)


    data['Occurrence_Date'] = pd.to_datetime(
    data['Occurrence_Date']).dt.time  # change data type
    data['Bike_Colour'].fillna('other', inplace=True)  # fill nan value
    data['Cost_of_Bike'].replace(0, np.nan, inplace=True)  # zero is also invalid

    unknown_make = ['UK', 'NULL', 'UNKNOWN MAKE', 'UNKNOWN', 'NONE', 'NO', 'UNKNOWNN',
                'UNKONWN', 'UNKOWN', 'UNKNONW', '-', 'UNKNOW', 'NO NAME', '?']  # all typos stand for known
    giant = data['Bike_Make'][data['Bike_Make'].str.contains(
     'giant', case=False, na=False)].unique().tolist()  # alias of giant
    giant.append('GI')

    data['Bike_Make'].replace(giant, 'GIANT', inplace=True)
    data['Bike_Make'].replace('OT', 'OTHER', inplace=True)
    data['Bike_Make'].replace(unknown_make, np.nan, inplace=True)

# transform non-numeric data
    encoder = preprocessing.LabelEncoder()
    data['Bike_Type'] = encoder.fit_transform(
    data['Bike_Type'])  # only numerical values for KNNImputer
    data['Bike_Make'] = pd.Series(encoder.fit_transform(data['Bike_Make'][data['Bike_Make'].notna(
    )]), index=data['Bike_Make'][data['Bike_Make'].notna()].index)  # only numerical values for KNNImputer
    data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']])
    data[['Bike_Type', 'Bike_Speed', 'Bike_Make']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Bike_Make']])

# Convert cost to cost catagory
    low = data['Cost_of_Bike'].quantile(.25)
    average = data['Cost_of_Bike'].quantile(.5)
    high = data['Cost_of_Bike'].quantile(.75)
    data['cost_catag'] = np.where(data['Cost_of_Bike'] <= low, 'low', np.where((data['Cost_of_Bike'] > low) & (
    data['Cost_of_Bike'] <= average), 'average', np.where((data['Cost_of_Bike'] > average) & (data['Cost_of_Bike'] <= high), 'high', 'luxury')))

# upcycling of data
    data['Status'].replace('STOLEN', 0, inplace=True)
    data['Status'].replace(['UNKNOWN', 'RECOVERED'], 1, inplace=True)

    print(data.head())

# encoding categorical features
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    for col in categorical_cols:
      data[col] = encoder.fit_transform(data[col])
    X, Y = data.drop('Status', axis=1), data['Status']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

    #clf_neural_network = MLPClassifier()
    clf_neural_network = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)

    # Neural Networks
    print(clf_neural_network)
    clf_neural_network.fit(x_train, y_train)
    clf_neural_network.score(x_train, y_train)
    y_pred = clf_neural_network.predict(x_test)
    print('Accuracy of NN is:', accuracy_score(y_test, y_pred))
    
    
    # Grid Search --- Neural Networks
    param_grid_mlp = {
        'activation': ['identity', 'logistic'],
        'solver': ['lbfgs', 'adam'],
        'alpha': [0.001, 0.01, 0.1],
        'learning_rate': ['constant']
    }
    grid_search_mlp = RandomizedSearchCV(
        estimator=clf_neural_network,
        param_distributions=param_grid_mlp,
        scoring='accuracy',
        cv=2,
        n_iter=5,
        refit=True,
        verbose=3
    )
    # Fitting the grid search
    grid_search_mlp.fit(x_train, y_train)
    
    # Printing the best parameters & estimator
    print("Best Params: ", grid_search_mlp.best_params_)
    print("Best estimators are: ", grid_search_mlp.best_estimator_)
    
    # Classification Report
    
    # Final Predictions Neural Network
    predictions_nn = grid_search_mlp.predict(x_test)
    model_accuracy_nn = metrics.accuracy_score(y_test, predictions_nn)
    print("Classification Report of NN:-\n ",
          metrics.classification_report(y_test, predictions_nn))
    print('Confusion Matrix(S): \n', confusion_matrix(y_test, predictions_nn))
    print('Accuracy Of NN', model_accuracy_nn)
    print('-------------------------')
    return accuracy_score(y_test, predictions_nn)
NeuralNetworksModel()


def randomForestModel():
    drop_column = ['OBJECTID', 'X', 'Y', 'event_unique_id', 'City',
                   'Location_Type', 'NeighbourhoodName', 'Latitude', 'Longitude', 'OBJECTID_1']

    data = pd.read_csv("J:\Centennial Stuff\Sem2\Supervised Learning\Project_Jayesh\Bicycle_Thefts.csv")

    X1 = data['Bike_Model']

    Y1 = data['Status']

    data.drop(drop_column, axis=1, inplace=True)
# Neighbourhood is identical with Hood ID
    data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True)

    data['Occurrence_Date'] = pd.to_datetime(
        data['Occurrence_Date']).dt.time  # change data type
    data['Bike_Colour'].fillna('other', inplace=True)  # fill nan value
    data['Cost_of_Bike'].replace(
        0, np.nan, inplace=True)  # zero is also invalid

    unknown_make = ['UK', 'NULL', 'UNKNOWN MAKE', 'UNKNOWN', 'NONE', 'NO', 'UNKNOWNN',
                    'UNKONWN', 'UNKOWN', 'UNKNONW', '-', 'UNKNOW', 'NO NAME', '?']  # all typos stand for known
    giant = data['Bike_Make'][data['Bike_Make'].str.contains(
        'giant', case=False, na=False)].unique().tolist()  # alias of giant
    giant.append('GI')

    data['Bike_Make'].replace(giant, 'GIANT', inplace=True)
    data['Bike_Make'].replace('OT', 'OTHER', inplace=True)
    data['Bike_Make'].replace(unknown_make, np.nan, inplace=True)

# transform non-numeric data
    encoder = preprocessing.LabelEncoder()
    data['Bike_Type'] = encoder.fit_transform(
        data['Bike_Type'])  # only numerical values for KNNImputer
    data['Bike_Make'] = pd.Series(encoder.fit_transform(data['Bike_Make'][data['Bike_Make'].notna(
    )]), index=data['Bike_Make'][data['Bike_Make'].notna()].index)  # only numerical values for KNNImputer
    data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Cost_of_Bike']])
    data[['Bike_Type', 'Bike_Speed', 'Bike_Make']] = KNNImputer(
    ).fit_transform(data[['Bike_Type', 'Bike_Speed', 'Bike_Make']])

# Convert cost to cost catagory
    low = data['Cost_of_Bike'].quantile(.25)
    average = data['Cost_of_Bike'].quantile(.5)
    high = data['Cost_of_Bike'].quantile(.75)
    data['cost_catag'] = np.where(data['Cost_of_Bike'] <= low, 'low', np.where((data['Cost_of_Bike'] > low) & (
        data['Cost_of_Bike'] <= average), 'average', np.where((data['Cost_of_Bike'] > average) & (data['Cost_of_Bike'] <= high), 'high', 'luxury')))

# upcycling of data
    data['Status'].replace('STOLEN', 0, inplace=True)
    data['Status'].replace(['UNKNOWN', 'RECOVERED'], 1, inplace=True)

    print(data.head())

# encoding categorical features
    categorical_cols = [
        col for col in data.columns if data[col].dtype == 'object']
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])
    X, Y = data.drop('Status', axis=1), data['Status']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(x_train, y_train)

    rf_y_pred = rf.predict(x_test)
    print('Accuracy of RandomForest is:', accuracy_score(y_test, rf_y_pred))

    # Random forest classifier with tuning
    params_grid = {
        'n_estimators': range(10, 100, 10),
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 10),
        'min_samples_leaf': range(1, 10),
    }

    rs = RandomizedSearchCV(
        rf,
        params_grid,
        n_iter=10,
        cv=10,
        scoring='accuracy',
        return_train_score=False,
        verbose=2,
        random_state=88)

    search = rs.fit(x_train, y_train)

    # best parameters & estimator
    print("Best Params: ", search.best_params_)
    print("Best estimators are: ", search.best_estimator_)

    accuracy = search.best_score_ * 100
    print(
        "Accuracy for training dataset with tuning is : {:.2f}%".format(accuracy))

    # Training the Random Forest Classification model on the Training Set with best param
    fine_tuned_model = search.best_estimator_.fit(x_train, y_train)

    # save model

    # Predicting the Test set results
    rf_y_pred = fine_tuned_model.predict(x_test)
    # predict_proba to return numpy array with two columns for a binary classification for N and P
    rf_y_scores = fine_tuned_model.predict_proba(x_test)

    # roc curve
    # fpr, tpr, threshold = roc_curve(y_test, rf_y_scores[:, 1])
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.title('ROC Curve of RandomForest')
    print('Classification Report(N): \n',
          classification_report(y_test, rf_y_pred))
    print('Confusion Matrix(N): \n', confusion_matrix(y_test, rf_y_pred))
    print('Accuracy(N): \n', metrics.accuracy_score(y_test, rf_y_pred))

    # Comparing the Real Values with Predicted Values
    df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': rf_y_pred})
    print(df)

    return metrics.accuracy_score(y_test, rf_y_pred)

randomForestModel()

################################################################