import os
import pandas as pd
import numpy as np
#import imblearn
import sys
import boto3
#import inline as inline
import requests 
#import json



s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')

projectID = sys.argv[4]
token =sys.argv[5]
processID = sys.argv[6]
try:
    name = sys.argv[1] ##new name of the output file
    inFile =(sys.argv[2]) ## name of the file to run outlier
    sdNO = int(sys.argv[3]) ## SD for the file

    df = pd.read_csv(inFile, index_col=False)

    colLen = len(df.columns)
    emptyRowSum = (df.isnull().sum(axis=1))
    df_EmptyRow = (((df.isnull().sum(axis=1))/len(df.columns)))*100
    df['Percentage'] =df_EmptyRow


    df.drop(df[df.Percentage > sdNO].index, inplace=True)
    df.drop('Percentage', axis=1, inplace=True)
    df.to_csv(name+'_No_Missing_data.csv')
    s3_resource.meta.client.upload_file(
        Filename=name+'_No_Missing_data.csv', Bucket='superlearner', Key=name+'_No_Missing_data.csv')
        
    key= name+'_No_Missing_data.csv'
    #key_two = name+'_Outlier_removed_graph.pdf'
    bucket = 'superlearner'
    New_url =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key}"
    #New_url_two =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_two}"


    # defining the api-endpoint  
    API_ENDPOINT = "https://file-ms-api.superlearnerscripts.com/process/complete_wone_dpp"
    payload = { "projectId": projectID,"processId": processID,"location": New_url, "name": key,
    "type": "csv"}
    headers = {'token': token}
    response = requests.post(API_ENDPOINT,  data = payload, headers= headers)
    
except Exception as ex:
    name = sys.argv[1]
    #key= name+'_No_Missing_data.csv'
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID, "projectId":projectID, "name":name, "reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)