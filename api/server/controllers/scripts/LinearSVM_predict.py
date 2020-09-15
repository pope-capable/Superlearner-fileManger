# set wd
import os
# Imports
import sys
import pickle 
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import urllib.request
import boto3
import inline as inline
import requests 
import json

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')

projectID = sys.argv[4]
token =sys.argv[5]
processID = sys.argv[6]

try:

    ### PRESCHOOL MODEL DEVELOPMENT ###
    # Import cleaned, unstandardised 4YR dataset

    inFile = sys.argv[3] # file to perform feature_selection on 
    Output_name = sys.argv[1]
    model = sys.argv[2]

    data_4YR = pd.read_csv(inFile, index_col=False)
    
    # 1368 Ids, 59 columns

    # Subset the 12 features selected from Balanced Random Forest RFE - 5-fold CV
    selected_data = data_4YR
    complete_data_4YR = selected_data.dropna()
    complete_data_4YR = complete_data_4YR.loc[:, ~complete_data_4YR.columns.str.contains('^Unnamed')]
  
    study_1 = complete_data_4YR.iloc[:,0]
    complete_subset_features_1 = complete_data_4YR.drop(complete_data_4YR.columns[[0]], axis=1)
    complete_subset_features = complete_data_4YR.drop(complete_data_4YR.columns[[0]], axis=1)
    Catergorical = complete_subset_features.select_dtypes(exclude=['number'])
    
    complete_subset_features = complete_subset_features[(list(complete_subset_features.columns))]

    le = preprocessing.LabelEncoder()

    complete_subset_features[Catergorical.columns] = complete_subset_features[Catergorical.columns].apply(le.fit_transform)

    
    


    best_clf = pickle.load(urllib.request.urlopen(model))

    #Predict the response for test dataset
    y_pred = best_clf.predict(complete_subset_features)

    complete_subset_features_1['Asthma_Outcome'] = y_pred 
    complete_subset_features_1 = pd.concat([study_1,complete_subset_features_1], axis=1)
    complete_subset_features_1.to_csv(Output_name+'LinearPredict.csv')

    

    s3_resource.meta.client.upload_file( 
    Filename=Output_name+'LinearPredict.csv',Bucket='superlearner',Key=Output_name+'LinearPredict.csv')
   
    key= Output_name+'LinearPredict.csv'
    
    bucket = 'superlearner'
    New_url =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key}"
  

        # defining the api-endpoint  
    API_ENDPOINT = "https://file-ms-api.superlearnerscripts.com/process/complete_wone"
    payload = { "projectId": projectID,"processId": processID,"location": New_url, "name": key,
    "type": "csv"}
    headers = {'token': token}
    response = requests.post(API_ENDPOINT,  data = payload, headers= headers)
    
except Exception as ex:
    name = sys.argv[1]
    #key= Output_name+'LinearPredict.csv'
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID, "projectId":projectID, "name":name, "reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)
    