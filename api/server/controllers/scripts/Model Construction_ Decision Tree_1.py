# Construct a Decision Tree classifier using data from individuals with complete data for all selected predictors and the outcome.

### Set working directory ###
import os
import sys
import pickle
from sklearn import preprocessing
# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score


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

    inFile = sys.argv[2] # file to perform feature_selection on
    study=sys.argv[3] #text box which is the outcome which is sys.arg[2] = Asthma_10YR, which should be entered in the textbox on the front end, that would display OUTCOME NAME 
    outc=sys.argv[4] #sys.arg[3] = Study_ID, the study group which should be entered in the textbox on the frontend,  which will display STUDY/INDEX NAME
    Output_name = sys.argv[1]

    ### Import and prepare data for model development ###
    # Import cleaned, unstandardised dataset containing all of the candidate predictors and outcome
    data_4YR = pd.read_csv(inFile, index_col=False)
    data_4YR = data_4YR.loc[:, ~data_4YR.columns.str.contains('^Unnamed')]

    # Subset the features selected from the feature selection method - RFE
    selected_data = data_4YR
    complete_data_4YR = selected_data.dropna()
    

    # Create data_features
    complete_subset_features =  complete_data_4YR.drop([outc], axis=1)
    

    # Create data_outcome
    complete_subset_outcome = complete_data_4YR[outc]

    Catergorical = complete_subset_features.select_dtypes(exclude=['number'])
    Binary=[col for col in complete_subset_features if np.isin(complete_subset_features[col].unique(), [0, 1]).all() ]
    ContinousVariable= complete_subset_features[complete_subset_features.columns.drop(Catergorical)]
    ContinousVariable= ContinousVariable[ContinousVariable.columns.drop(Binary)]
    ContinousVariable= list(ContinousVariable.columns)
    
    complete_subset_features = complete_subset_features[(list(complete_subset_features.columns))]

    le = preprocessing.LabelEncoder()

    complete_subset_features[Catergorical.columns] = complete_subset_features[Catergorical.columns].apply(le.fit_transform)

    

    # Split dataset into training set and test set: 2:1 ratio, stratified by outcome to preserve the orginial class balance in the two splits
    X_train, X_test, y_train, y_test = train_test_split(complete_subset_features, complete_subset_outcome,
                                                        stratify=complete_subset_outcome, 
                                                        test_size=0.333, shuffle=True, random_state=123)
                                                        
    ### Standardise training and test sets
    # Identify continuous features
    #ContinousVariable = complete_data_4YR.select_dtypes(exclude=['number'])
    TrainingSet = pd.concat([X_train,y_train], axis=1)
    TestSet = pd.concat([X_test,y_test], axis=1)
    X_test = X_test.drop([study],axis=1)
    X_train = X_train.drop([study],axis=1)
    ContinousVariable.remove(study)
    #complete_subset_features =  complete_data_4YR.drop([study, outc], axis=1)

    TrainingSet.to_csv(Output_name+'TrainingSet.csv')
    TestSet.to_csv(Output_name+'TestSet.csv')

    cont = X_train[ContinousVariable]
    scaler = StandardScaler()
    # Standardise continuous features in the training set
    cont_train = pd.DataFrame(scaler.fit_transform(X_train.iloc[:,0:len(ContinousVariable)]), columns=(ContinousVariable))
    cat_train = X_train.iloc[:,len(ContinousVariable):]
    SX_train = pd.concat([cont_train, cat_train.reset_index(drop=True)], axis=1)
    # Standardise continuous features in the test set
    cont_test = pd.DataFrame(scaler.transform(X_test.iloc[:,0:len(ContinousVariable)]), columns=(ContinousVariable))
    cat_test = X_test.iloc[:,len(ContinousVariable):]
    SX_test = pd.concat([cont_test, cat_test.reset_index(drop=True)], axis=1)
    
    #########################################
    ### Model construction: Decision Tree ###
    #########################################
    #Define the classifier
    clf = DecisionTreeClassifier(random_state=123)	

    ##### Grid search #####
    # Define a narrowed search range for each hyperparameter
    sample_split=np.arange(2,12,1)
    param_dist = {"max_depth": [1,2,3,4,5,6,7,8,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,None],
                  "splitter": ['best', 'random'],
                  "max_features": ['log2', 'sqrt', 'auto', None],
                  "min_samples_split": sample_split,
                  "criterion": ["gini","entropy"]}

    # Run grid search within 5-fold cross validation - scoring parameter = measure on which to evaluate the optimal model performance
    grid_search = GridSearchCV(clf, scoring='balanced_accuracy', param_grid=param_dist, cv=StratifiedKFold(5), n_jobs=1)
    start = time()
    grid_search.fit(SX_train, y_train)
    
    # Record time taken for grid search
    GStime = (time() - start)
    print (GStime)

    # Print the hyperpameters and balanced accuracy score that provided the best model performance from the grid search 
    best_parameters = grid_search.best_params_
    print(best_parameters)
    best_score = grid_search.best_score_
    print(best_score)

    # Save grid search results if desired:
    GSresults = pd.DataFrame(grid_search.cv_results_)
    filename = Output_name+"_filename.csv"
    GSresults.to_csv(filename,index=False)

    # Redefine the classifier with optimal hyperparameters
    #best_clf = DecisionTreeClassifier(criterion='gini', max_depth=10, max_features='log2', min_samples_split=4, splitter='random', random_state=123)
    best_clf = DecisionTreeClassifier(criterion=best_parameters["criterion"], max_depth=best_parameters["max_depth"], max_features=best_parameters["max_features"], min_samples_split=best_parameters["min_samples_split"], splitter=best_parameters["splitter"], random_state=123)	
    filename2 = Output_name+'DecisionTree_fitted_model.mod'
    with open( filename2, "wb") as file:
        pickle.dump(best_clf, file)

                    
    # Fit optimised model
    best_clf.fit(SX_train,y_train)

    filename = Output_name+'DecisionTree_finalized_model.mod'
    with open( filename, "wb") as file:
        pickle.dump(best_clf, file)


    ### Evalaute performance on the training set ###
    y_train_pred = best_clf.predict(SX_train)

    # Print confusion matrix 
    cm_train = confusion_matrix(y_train, y_train_pred)
    print(cm_train)


    train_report = classification_report(y_train, y_train_pred)
    print (train_report)

    balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
    accuracy_train = balanced_accuracy

    sensitivity =  cm_train[1,1]/(cm_train[1,0]+cm_train[1,1])								
    print(sensitivity)

    specificity = cm_train[0,0]/(cm_train[0,0]+cm_train[0,1])									
    print(specificity)

    PPV = cm_train[1,1]/(cm_train[1,1]+cm_train[0,1])	
    print(PPV)

    NPV = cm_train[0,0]/(cm_train[0,0]+cm_train[1,0])
    print(NPV)

    LRp = sensitivity/(1-specificity)
    print(LRp)

    LRn = (1-sensitivity)/specificity
    print(LRn)

    #  AUC: 
    probs = best_clf.predict_proba(SX_train)
    preds = probs[:,1]
    ROCAUC_train = roc_auc_score(y_train, preds)
    print(ROCAUC_train)




    file1 = open(Output_name+'_Decision Tree_Model_Report.txt', 'w') 

    L = ["Grid search best score \n",(str(best_score))+"\n\n\n",  "Confusion Matrix \n"+(str(cm_train))+" \n\n\n", 
        "Train report"+(str(train_report))+" \n\n\n", "Accuracy score \n"+(str(accuracy_train))+"\n\n\n", "Sensitivity \n"+(str(sensitivity))+"\n\n\n", 
        "Specificity \n"+(str(specificity))+"\n\n\n", "PPV \n"+ (str(PPV)) +"\n\n\n","NPV \n"+(str(NPV))+"\n\n\n", "LRP \n"+(str(LRp))+" \n\n\n", 
        "LRN \n"+(str(LRn))+" \n\n\n" , "ROCAUC_train"+(str(ROCAUC_train))+" \n\n\n"] 



    ### Predict the outcome in the test set and evaluate performance 
    y_pred = best_clf.predict(SX_test)

    # Print confusion matrix
    cm_test = confusion_matrix(y_test, y_pred)	
    print (cm_test)

    test_report = classification_report(y_test, y_pred)
    print (test_report)

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    accuracy_test = balanced_accuracy

    sensitivity_test =  cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])								
    #print(sensitivity)

    specificity_test = cm_test[0,0]/(cm_test[0,0]+cm_test[0,1])									
    #print(specificity)

    PPV_test = cm_test[1,1]/(cm_test[1,1]+cm_test[0,1])	
    #print(PPV)

    NPV_test = cm_test[0,0]/(cm_test[0,0]+cm_test[1,0])
    #print(NPV)

    LRp_test = sensitivity_test/(1-specificity_test)
    print(LRp)

    LRn_test = (1-sensitivity_test)/specificity_test
    print(LRn)

    probs = best_clf.predict_proba(SX_test)
    preds = probs[:,1]
    ROCAUC_test = roc_auc_score(y_test, preds)
    print(ROCAUC_test)


    testy = ["Confusion Matrix \n"+(str(cm_test))+" \n\n\n", "Test report"+(str(test_report))+" \n\n\n", 
    "Accuracy score \n"+(str(accuracy_test))+"\n\n\n", "Sensitivity \n"+(str(sensitivity_test))+"\n\n\n", 
    "Specificity \n"+(str(specificity_test))+"\n\n\n", "PPV \n"+ (str(PPV_test)) +"\n\n\n",
    "NPV \n"+(str(NPV_test))+"\n\n\n", "LRP \n"+(str(LRp_test))+" \n\n\n", "LRN \n"+(str(LRn_test))+" \n\n\n" , 
    "ROCAUC_test"+(str(ROCAUC_test))+" \n\n\n"] 

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
        Filename=Output_name+'_Decision Tree_Model_Report.txt',Bucket='superlearner',Key=Output_name+'_Decision Tree_Model_Report.txt')
    s3_resource.meta.client.upload_file( 
        Filename=Output_name+'TrainingSet.csv',Bucket='superlearner',Key=Output_name+'TrainingSet.csv')
    s3_resource.meta.client.upload_file(
        Filename=Output_name+'TestSet.csv',Bucket='superlearner',Key=Output_name+'TestSet.csv')

    key= filename
    key_two =Output_name+'_Decision Tree_Model_Report.txt'
    key_three =Output_name+'TrainingSet.csv'
    key_four =Output_name+'TestSet.csv'
    key_five= filename2
    bucket = 'superlearner'
    New_url =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key}"
    New_url_two =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_two}"
    New_url_three =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_three}"
    New_url_four =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_four}"
    New_url_five =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_five}"




        # defining the api-endpoint  
    API_ENDPOINT = "https://file-ms-api.superlearnerscripts.com/process/complete_model"
    #payload = { "projectId": projectID,"processId": processID,"location": New_url, "locationTwo":New_url_two, "modelName": key, "nameTwo":key_two, "modelType": "Decision_Tree", "typeTwo":"txt"}
    payload = { "projectId": projectID,"processId": processID,"location": New_url, "locationTwo":New_url_two, "locationThree":New_url_three, "locationFour":New_url_four, "locationFive":New_url_five, "modelName": key,
    "nameTwo":key_two, "nameThree":key_three, "nameFour":key_four,  "modelType": "Decision_Tree", "typeTwo":"txt","typeThree":"csv","typeFour":"csv"}

    headers = {'token': token}
    response = requests.post(API_ENDPOINT,  data = payload, headers= headers)
    
except Exception as ex:
    name = sys.argv[1]
    key = name
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID, "projectId":projectID, "name":key, "reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)
