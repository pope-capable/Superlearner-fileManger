##############
### boruta ###
##############

import os
import sys
# Import package
from boruta import BorutaPy
import pandas as pd
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import preprocessing
import boto3
import requests

### Set working directory ###
##os.chdir("")

### Import and prepare data for RFE
# Import cleaned, unstandardised dataset
s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')

projectID = sys.argv[5] #req.body.projectId,
token =sys.argv[6] # req.headers.token,
processID = sys.argv[7] # processIdentifier,

try:
    filenamee=sys.argv[1] #new name for the file output
    inFile = sys.argv[2] # file to perform feature_selection on
    study=sys.argv[3] #sys.arg[3] = Study_ID, the study group which should be entered in the textbox on the frontend,  which will display STUDY/INDEX NAME
    outc=sys.argv[4] #text box which is the outcome which is sys.arg[2] = Asthma_10YR, which should be entered in the textbox on the front end, that would display OUTCOME NAME 



    File = pd.read_csv(inFile, index_col=False)

    # Remove individuals with missing data to create a subset of individuals with only complete data
    New_File = File.dropna()
    New_File.drop(New_File.filter(regex="Unname"),axis=1, inplace=True)

    study_1 = New_File[study]
    outc_1 = New_File[outc]

    outc_Y=New_File[[outc]]
    dataset_X=New_File.drop([outc] , axis='columns')
    fea=dataset_X
    dataset_X=dataset_X.drop([study], axis=1)
    Catergorical = dataset_X.select_dtypes(exclude=['number'])
	
    Binary=[col for col in dataset_X if np.isin(dataset_X[col].unique(), [0, 1]).all() ]
    ContinousVariable= dataset_X[dataset_X.columns.drop(Catergorical)]
    ContinousVariable= ContinousVariable[ContinousVariable.columns.drop(Binary)]
    Catergorical_1 = [dataset_X.columns.get_loc(processID) for processID in list(Catergorical.columns) if processID in dataset_X]
	#Catergorical_1 = column_index(Catergorical, [Catergorical])
    Binary = New_File[Binary]

    dataset_X =  dataset_X[dataset_X.columns.drop(ContinousVariable)]

    Xs = pd.concat([ContinousVariable,dataset_X], axis=1)
	
    best_param1= {'bootstrap': True,'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 100}
    # Used a balanced random forest classifier to account for the class imbalance in the dataset
    bclf = BalancedRandomForestClassifier(n_estimators=best_param1["n_estimators"],max_depth=best_param1["max_depth"], min_samples_split =best_param1["min_samples_split"],max_features=best_param1["max_features"],random_state=123)
    # Define feature selection method
    boruta = BorutaPy(bclf, verbose=2, n_estimators=100, random_state=123)

    pipeline = Pipeline([
        ('standardising', Pipeline([
            ('select', ColumnTransformer([
                ('scale', StandardScaler(),list(ContinousVariable.columns))
                ],
                remainder='passthrough')
            )
        ])),
       ('bclf', boruta) 
    ])

    # apply Boruta to data
    #X=X.to_numpy()

    Xs = Xs[(list(Xs.columns))]

    le = preprocessing.LabelEncoder()

    Xs[Catergorical.columns] = Xs[Catergorical.columns].apply(le.fit_transform)
	#X[Catergorical.columns] = le.fit_transform(list(Catergorical.columns))
    fit = pipeline.fit(Xs, outc_Y.values.ravel())

    # Check selected features 
    boruta.support_
    print(boruta.support_)

    #check ranking of features
    S = boruta.ranking_
    Selected = pd.DataFrame(S)
    Features = New_File.iloc[:,1:New_File.shape[1]-1]
    names = pd.DataFrame(Features.columns)
    list = pd.concat([names, Selected], axis=1)
    list.columns=['Feature', 'Boruta_ranking']
    #call transform() on X to filter it down to selected features
    X_filtered = boruta.transform(Xs.values)
    X_filtered = list.loc[(list['Boruta_ranking'] <= 15)] 
    X_filtered.to_csv(filenamee+'_Filtered_boruta_FS.csv')
    df = pd.DataFrame(New_File)
    df = df.loc[:, X_filtered['Feature']]
    df = pd.concat([study_1,df,outc_1], axis=1)
    df.to_csv(filenamee+'_Feature_selected.csv')

    X_filtered1 = DataFrame(X_filtered,columns=['Feature','Boruta_ranking'])
    X_filtered1.plot(x ='Feature', y='Boruta_ranking', kind = 'line')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Boruta_Ranking')
    plt.xlabel('Feature selected')
    plt.tight_layout()
    plt.savefig(filenamee+'_Filtered_boruta_FS.pdf')
    s3_resource.meta.client.upload_file( 
        Filename=filenamee+'_Filtered_boruta_FS.csv',Bucket='superlearner',Key=filenamee+'_Filtered_boruta_FS.csv')
    s3_resource.meta.client.upload_file(
        Filename=filenamee+'_Feature_selected.csv',Bucket='superlearner',Key=filenamee+'_Feature_selected.csv')
    
    key= filenamee+'_Filtered_boruta_FS.csv'
    key_two =filenamee+'_Feature_selected.csv'
    bucket = 'superlearner'
    New_url =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key}"
    New_url_two =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_two}"


        # defining the api-endpoint  
    API_ENDPOINT = "https://file-ms-api.superlearnerscripts.com/process/complete_dpp"
    payload = { "projectId": projectID,"processId": processID,"location": New_url, "location_two":New_url_two, "name": key, "name_two":key_two,
    "type": "csv", "type_two":"csv"}
    headers = {'token': token}
    response = requests.post(API_ENDPOINT,  data = payload, headers= headers)

except Exception as ex:
    name = sys.argv[1]
    #key= filenamee+'_Filtered_boruta_FS.csv'
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID, "projectId":projectID, "name":name, "reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)