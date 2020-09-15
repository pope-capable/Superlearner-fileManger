# Import packages
import os
import sys
import boto3

import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests 
import json
  

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')

projectID = sys.argv[4]
token =sys.argv[5]
processID = sys.argv[6]

try:
    name = sys.argv[1] ##new name of the output file
    inFile =(sys.argv[2]) ## name of the file to run outlier
    sdNO = int(sys.argv[3]) ## SD for the file
   
    #d = sys.argv[7]
    #e = sys.argv[8]

    New_File = pd.read_csv(inFile, index_col=False)

    # Remove individuals with missing data to create a subset of individuals with only complete data
    #New_File = File.drop(File.columns[0], axis=1)
    New_File.drop(New_File.filter(regex="Unname"),axis=1, inplace=True)
    #New_File.to_csv('sdsdsdsdsdsd.csv')

    Catergorical = New_File.select_dtypes(exclude=['number'])
	
    Binary=[col for col in New_File if np.isin(New_File[col].unique(), [0, 1]).all() ]
    ContinousVar= New_File[New_File.columns.drop(Catergorical)]
    ContinousVar= ContinousVar[ContinousVar.columns.drop(Binary)]

    
    #ContinousVar= New_File.drop(Binary_Var_Col, axis=1)
    Binary_V = New_File.drop(ContinousVar, axis = 1)
    ##print(ContinousVar)
    Study_ID=ContinousVar.iloc[:,0]
    
    #print(Study_ID)
    ContinousVar= ContinousVar.drop(ContinousVar.columns[0], axis=1)
    ##print(ContinousVar)

    ContinousVar_NO_Outlier =ContinousVar.mask(ContinousVar.sub(ContinousVar.mean()).div(ContinousVar.std()).abs().gt(sdNO))
    ##print(ContinousVar_NO_Outlier)
  

    ##plt.boxplot(ContinousVar_NO_Outlier.columns.values)
    boxplot = ContinousVar_NO_Outlier.boxplot(grid=False, rot=55, fontsize=25)
    plt.tight_layout()
    plt.savefig(name+'_Outlier_removed_graph.pdf')
    #plt.show()
    File_without_Outlier = pd.concat([Study_ID,ContinousVar_NO_Outlier,Binary_V], axis=1)
    #File_without_Outlier = File_without_Outlier.loc[:, ~File_without_Outlier.columns.str.contains('^Unnamed')]
    File_without_Outlier.drop(File_without_Outlier.columns[File_without_Outlier.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    File_without_Outlier.to_csv(name+'_No_Outlier.csv')
    s3_resource.meta.client.upload_file(
        Filename=name+'_No_Outlier.csv', Bucket='superlearner', Key=name+'_No_Outlier.csv')
    s3_resource.meta.client.upload_file(
        Filename=name+'_Outlier_removed_graph.pdf', Bucket='superlearner', Key=name+'_Outlier_removed_graph.pdf')
        
    key= name+'_No_Outlier.csv'
    key_two = name+'_Outlier_removed_graph.pdf'
    bucket = 'superlearner'
    New_url =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key}"
    New_url_two =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_two}"


        # defining the api-endpoint  
    API_ENDPOINT = "https://file-ms-api.superlearnerscripts.com/process/complete_dpp"
    payload = { "projectId": projectID,"processId": processID,"location": New_url, "location_two":New_url_two, "name": key, "name_two":key_two,
    "type": "csv", "type_two":"pdf"}
    headers = {'token': token}
    response = requests.post(API_ENDPOINT,  data = payload, headers= headers)
except Exception as ex:
    name = sys.argv[1]
    key= name
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID,"projectId":projectID, "name":key,"reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)

    


