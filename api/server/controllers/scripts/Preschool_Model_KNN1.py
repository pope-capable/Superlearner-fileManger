# set wd
import os
import sys
import pickle
import joblib
from sklearn import preprocessing
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import boto3
import inline as inline
import requests 
import json

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
np.random.seed(123)

projectID = sys.argv[5]
token =sys.argv[6]
processID = sys.argv[7]

try:
    ### PRESCHOOL MODEL DEVELOPMENT ###
    # Import cleaned, unstandardised 4YR dataset
    inFile = sys.argv[2] # file to perform feature_selection on
    study=sys.argv[3] #text box which is the outcome which is sys.arg[2] = Asthma_10YR, which should be entered in the textbox on the front end, that would display OUTCOME NAME 
    outc=sys.argv[4] #sys.arg[3] = Study_ID, the study group which should be entered in the textbox on the frontend,  which will display STUDY/INDEX NAME
    Output_name= sys.argv[1]

    data_4YR = pd.read_csv(inFile, index_col=False)
    

    # Subset the 12 features selected from Balanced Random Forest RFE - 5-fold CV
    selected_data = data_4YR
    complete_data_4YR = selected_data.dropna()
    complete_data_4YR = complete_data_4YR.loc[:, ~complete_data_4YR.columns.str.contains('^Unnamed')]
    # n=548

    #create data_features
    complete_subset_features = complete_data_4YR.drop([outc], axis=1)

    #create data_outcome
    complete_subset_outcome = complete_data_4YR[outc]

    Catergorical = complete_subset_features.select_dtypes(exclude=['number'])
    Binary=[col for col in complete_subset_features if np.isin(complete_subset_features[col].unique(), [0, 1]).all() ]
    ContinousVariable= complete_subset_features[complete_subset_features.columns.drop(Catergorical)]
    ContinousVariable= ContinousVariable[ContinousVariable.columns.drop(Binary)]
    ContinousVariable= list(ContinousVariable.columns)
    
    complete_subset_features = complete_subset_features[(list(complete_subset_features.columns))]

    le = preprocessing.LabelEncoder()

    complete_subset_features[Catergorical.columns] = complete_subset_features[Catergorical.columns].apply(le.fit_transform)

    

    # Split dataset into training set and test set: 2/3 training and 1/3 test
    X_train, X_test, y_train, y_test = train_test_split(complete_subset_features, complete_subset_outcome,
                                                        stratify=complete_subset_outcome, 
                                                        test_size=0.333, shuffle=True, random_state=123)
                                                        
    TrainingSet = pd.concat([X_train,y_train], axis=1)
    TestSet = pd.concat([X_test,y_test], axis=1)
    X_test = X_test.drop([study],axis=1)
    X_train = X_train.drop([study],axis=1)
    ContinousVariable.remove(study)
    #complete_subset_features =  complete_data_4YR.drop([study, outc], axis=1)

    TrainingSet.to_csv(Output_name+'TrainingSet.csv')
    TestSet.to_csv(Output_name+'TestSet.csv')
    # Training set (n=365, asthma=51, no asthma=314)	Test set (n=183, asthma=25, no asthma=158)

    # Standardise training and test sets
    
    cont = X_train[ContinousVariable]

    scaler = StandardScaler()
    cont_train = pd.DataFrame(scaler.fit_transform(X_train.iloc[:,0:len(ContinousVariable)]), columns=(ContinousVariable))
    cat_train = X_train.iloc[:,len(ContinousVariable):]
    SX_train = pd.concat([cont_train, cat_train.reset_index(drop=True)], axis=1)

    cont_test = pd.DataFrame(scaler.transform(X_test.iloc[:,0:len(ContinousVariable)]), columns=(ContinousVariable))
    cat_test = X_test.iloc[:,len(ContinousVariable):]
    SX_test = pd.concat([cont_test, cat_test.reset_index(drop=True)], axis=1)



    #########################
    ### Define classifier ###
    #########################
    knn = KNeighborsClassifier()

    #####################
    #### Grid search ####
    #####################
    clf = KNeighborsClassifier()

    k_range = np.arange(1, 100, 1)
    p_range=[1,2]
    weights=['uniform', 'distance']
    param_grid=dict(n_neighbors=k_range, p=p_range, weights=weights)

    grid_search = GridSearchCV(clf, scoring='balanced_accuracy', param_grid=param_grid, cv=StratifiedKFold(5), n_jobs=1)
    start = time()
    grid_search.fit(SX_train, y_train)
    GStime = (time() - start)

    # best parameters
    best_parameters = grid_search.best_params_
    print(best_parameters)
    #{'n_neighbors': 9, 'p': 1, 'weights': 'uniform'}

    best_score = grid_search.best_score_
    print(best_score)
    # 0.6309978963933891

    ###############################
    #### Re-run optimised model ###
    ###############################
    # Define otpimised classifier
    best_clf = KNeighborsClassifier(n_neighbors=best_parameters["n_neighbors"], p=best_parameters["p"], weights=best_parameters["weights"])
    filename2 = Output_name+'_Knn_fitted_model.mod'
    with open( filename2, "wb") as file:
        pickle.dump(best_clf, file)

    # Train the model using the training sets
    scores = cross_val_score(best_clf, SX_train, y_train, scoring='balanced_accuracy', n_jobs=1, cv=StratifiedKFold(5))
    accuracy = scores.mean()
    sd = (scores.std())      
    # 0.8876619770455386 +/-  0.01601723122333163

    # Fit optimised model
    best_clf.fit(SX_train,y_train)

    filename = Output_name+'_Knn_finalized_model.mod'
    with open( filename, "wb") as file:
        pickle.dump(best_clf, file)
    ### Training set Performance
    y_train_pred = best_clf.predict(SX_train)
    cm_train = confusion_matrix(y_train, y_train_pred)
    print(cm_train)
    #[312   2] [ 35  16]


    train_report = classification_report(y_train, y_train_pred)
   
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print(accuracy_score)
    # 0.8986301369863013

    sensitivity =  cm_train[1,1]/(cm_train[1,0]+cm_train[1,1])								
    print(sensitivity)
    #0.3137254901960784

    specificity = cm_train[0,0]/(cm_train[0,0]+cm_train[0,1])									
    print(specificity)
    #0.9936305732484076

    PPV = cm_train[1,1]/(cm_train[1,1]+cm_train[0,1])	
    print(PPV)
    #0.8888888888888888

    NPV = cm_train[0,0]/(cm_train[0,0]+cm_train[1,0])
    print(NPV)
    #0.899135446685879

    LRp = sensitivity/(1-specificity)
    print(LRp)
    #49.25490196078432

    LRn = (1-sensitivity)/specificity
    print(LRn)
    #0.6906737053795877

    #  AUC: 
    probs = best_clf.predict_proba(SX_train)
    preds = probs[:,1]
    ROCAUC_train = roc_auc_score(y_train, preds)
    print(ROCAUC_train)
    # 0.9063007368552517

    file1 = open(Output_name+'_KNN_Model_Report.txt', 'w') 

    L = ["Grid search best score \n",(str(best_score))+"\n\n\n", "Sd of score \n"+(str(sd))+" \n\n\n", "Confusion Matrix \n"+(str(cm_train))+" \n\n\n", "Train report"+(str(train_report))+" \n\n\n", "Accuracy score \n"+(str(accuracy_train))+"\n\n\n", "Sensitivity \n"+(str(sensitivity))+"\n\n\n", "Specificity \n"+(str(specificity))+"\n\n\n", "PPV \n"+ (str(PPV)) +"\n\n\n","NPV \n"+(str(NPV))+"\n\n\n", "LRP \n"+(str(LRp))+" \n\n\n", "LRN \n"+(str(LRn))+" \n\n\n" , "ROCAUC_train"+(str(ROCAUC_train))+" \n\n\n"] 


    #Predict the response for test dataset
    y_pred = best_clf.predict(SX_test)

    cm_test = confusion_matrix(y_test, y_pred)	
    print (cm_test)
    # [[153   5] [ 21   4]

    test_report = classification_report(y_test, y_pred)
    print (test_report)

    accuracy_test = accuracy_score(y_test, y_pred)
    #0.8579234972677595

    sensitivity_test =  cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])								
    print(sensitivity)
    #0.16

    specificity_test = cm_test[0,0]/(cm_test[0,0]+cm_test[0,1])									
    print(specificity)
    #0.9683544303797469

    PPV_test = cm_test[1,1]/(cm_test[1,1]+cm_test[0,1])	
    print(PPV)
    #0.4444444444444444

    NPV_test = cm_test[0,0]/(cm_test[0,0]+cm_test[1,0])
    print(NPV)
    #0.8793103448275862

    LRp_test = sensitivity/(1-specificity)
    print(LRp)
    #5.056000000000009

    LRn_test = (1-sensitivity)/specificity
    print(LRn)
    #0.8674509803921567

    probs = best_clf.predict_proba(SX_test)
    preds = probs[:,1]
    ROCAUC_test = roc_auc_score(y_test, preds)
    print(ROCAUC_test)
    #0.7827848101265822


    testy = ["Confusion Matrix \n"+(str(cm_test))+" \n\n\n", "Test report"+(str(test_report))+" \n\n\n", "Accuracy score \n"+(str(accuracy_test))+"\n\n\n", "Sensitivity \n"+(str(sensitivity_test))+"\n\n\n", "Specificity \n"+(str(specificity_test))+"\n\n\n", "PPV \n"+ (str(PPV_test)) +"\n\n\n","NPV \n"+(str(NPV_test))+"\n\n\n", "LRP \n"+(str(LRp_test))+" \n\n\n", "LRN \n"+(str(LRn_test))+" \n\n\n" , "ROCAUC_test"+(str(ROCAUC_test))+" \n\n\n"] 

    file1.write("TRAINED MODEL EVALUATION REPORT \n\n\n\n\n") 
    file1.writelines(L) 
    file1.write("TEST/PREDICTED EVALUATION REPORT \n\n\n\n\n")
    file1.writelines(testy) 
    # Closing file 
    file1.close()
    
    s3_resource.meta.client.upload_file( 
        Filename=filename,Bucket='superlearner',Key=filename)
    s3_resource.meta.client.upload_file( 
        Filename=filename2,Bucket='superlearner',Key=filename2)
    s3_resource.meta.client.upload_file(
        Filename=Output_name+'_KNN_Model_Report.txt',Bucket='superlearner',Key=Output_name+'_KNN_Model_Report.txt')
    s3_resource.meta.client.upload_file( 
        Filename=Output_name+'TrainingSet.csv',Bucket='superlearner',Key=Output_name+'TrainingSet.csv')
    s3_resource.meta.client.upload_file(
        Filename=Output_name+'TestSet.csv',Bucket='superlearner',Key=Output_name+'TestSet.csv')

    
    key= filename
    key_two = Output_name+'_KNN_Model_Report.txt'
    key_three =Output_name+'TrainingSet.csv'
    key_four =Output_name+'TestSet.csv'
    key_five = filename2
    bucket = 'superlearner'
    New_url =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key}"
    New_url_two =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_two}"
    New_url_three =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_three}"
    New_url_four =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_four}"
    New_url_five =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_five}"



        # defining the api-endpoint  
    API_ENDPOINT = "https://file-ms-api.superlearnerscripts.com/process/complete_model"
    #payload = { "projectId": projectID,"processId": processID,"location": New_url, "locationTwo":New_url_two, "modelName": key, "nameTwo":key_two, "modelType": "knn_Model", "typeTwo":"txt"}

    payload = { "projectId": projectID,"processId": processID,"location": New_url, "locationTwo":New_url_two, "locationThree":New_url_three, "locationFour":New_url_four,"locationFive":New_url_five, "modelName": key,
    "nameTwo":key_two, "nameThree":key_three, "nameFour":key_four, "modelType": "knn_Model", "typeTwo":"txt","typeThree":"csv","typeFour":"csv"}

    headers = {'token': token}
    response = requests.post(API_ENDPOINT,  data = payload, headers= headers)
except Exception as ex:
    name = sys.argv[1]
    key= name
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID,"projectId":projectID, "name":key, "reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)