# set wd
import os
from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn import preprocessing
# Imports
import sys
import pickle 
import pandas as pd
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

projectID = sys.argv[5]
token =sys.argv[6]
processID = sys.argv[7]


try:
    ### PRESCHOOL MODEL DEVELOPMENT ###
    # Import cleaned, unstandardised 4YR dataset
    inFile = sys.argv[4] # file to perform feature_selection on 
    Output_name = sys.argv[1]
    models = sys.argv[3]
    mod = sys.argv[2]

    data_4YR = pd.read_csv(inFile, index_col=False)

    # 1368 Ids, 59 columns

    # Subset the 12 features selected from Balanced Random Forest RFE - 5-fold CV
    selected_data = data_4YR
    complete_data_4YR = selected_data.dropna()
    complete_data_4YR = complete_data_4YR.loc[:, ~complete_data_4YR.columns.str.contains('^Unnamed')]

    study_1 = complete_data_4YR.iloc[:,0]
    complete_subset_features = complete_data_4YR.drop(complete_data_4YR.columns[[0]], axis=1)
    complete_subset_features2 = complete_subset_features
    Catergorical = complete_subset_features.select_dtypes(exclude=['number'])
    Binary=[col for col in complete_subset_features if np.isin(complete_subset_features[col].unique(), [0, 1]).all() ]
    ContinousVariable= complete_subset_features[complete_subset_features.columns.drop(Catergorical)]
    ContinousVariable= ContinousVariable[ContinousVariable.columns.drop(Binary)]
    ContinousVariable= list(ContinousVariable.columns)
    
    complete_subset_features = complete_subset_features[(list(complete_subset_features.columns))]

    le = preprocessing.LabelEncoder()

    complete_subset_features[Catergorical.columns] = complete_subset_features[Catergorical.columns].apply(le.fit_transform)


    print(study_1)
    meta_model = pickle.load(urllib.request.urlopen(models))
    print(meta_model)


    modelss = []
    for arg in mod.split(","):
        modelss.append(pickle.load(urllib.request.urlopen(arg)))
    #modelss = modelss.split(",")
   # modelss = modelss.replace("[", "")
   # modelss = modelss.replace("]", " ")



    ##def get_model():
        #model = list()
        #model = list()
        #model = []
     ##   for arg in mod.split(","):
         #   model.append(pickle.load(urllib.request.urlopen(arg)))
        #return model
        

    def super_learner_predictions(X, modelT, meta_model):
        meta_X =[]
        for mode in modelT:
           # mode = [modelT.split("),") for modelT in mode]
            print('charlie',mode)
            yhat = mode.predict_proba(X)
            meta_X.append(yhat)
        meta_X = hstack(meta_X)
        return meta_model.predict(meta_X)
        
    #modelss =get_model()
    
     
    modelss = modelss[0]
    print('champopm',modelss)

    #Predict the response for test dataset
    y_pred = super_learner_predictions(complete_subset_features, modelss, meta_model)
    #print('qwiqwuw',y_pred)
    complete_subset_features2['Asthma_Outcome'] = y_pred 
    complete_subset_features2 = pd.concat([study_1,complete_subset_features2], axis=1)
    complete_subset_features2.to_csv(Output_name+'_Super_Predict.csv')


    s3_resource.meta.client.upload_file( 
        Filename=Output_name+'_Super_Predict.csv',Bucket='superlearner',Key=Output_name+'_Super_Predict.csv')
   
    key= Output_name+'_Super_Predict.csv'
    
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
    key= name
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID, "projectId":projectID, "name":key,"reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)
    