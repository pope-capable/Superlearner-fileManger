from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import sys
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
# Import packages
import pandas as pd
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
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
import boto3
import inline as inline
import urllib.request
import requests 
import json

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')

projectID = sys.argv[3]
token =sys.argv[4]
processID = sys.argv[5]

try:

    inFile = sys.argv[2] # file to perform feature_selection on
    study=sys.argv[6] #text box which is the outcome which is sys.arg[2] = Asthma_10YR, which should be entered in the textbox on the front end, that would display OUTCOME NAME 
    outc=sys.argv[7] #sys.arg[3] = Study_ID, the study group which should be entered in the textbox on the frontend,  which will display STUDY/INDEX NAME
    Output_name = sys.argv[1]
    modeling = sys.argv[8]

    modeling = modeling.replace("[", "")
    modeling = modeling.replace("]", " ")
    modeling = modeling.replace("'", "")
    modeling = modeling.split(",")

    def get_model():
        model = list()
        for arg in modeling:
            model.append(pickle.load(urllib.request.urlopen(arg)))
        
        return model
   

    ### Import and prepare data for model development ###
    # Import cleaned, unstandardised dataset containing all of the candidate predictors and outcome
    data_4YR = pd.read_csv(inFile, index_col=False)
    data_4YR = data_4YR.loc[:, ~data_4YR.columns.str.contains('^Unnamed')]

    # Subset the features selected from the feature selection method - RFE
    selected_data = data_4YR
    complete_data_4YR = selected_data.dropna()

    # Create data_features
    complete_subset_features =  complete_data_4YR.drop([study, outc], axis=1)

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




    X = complete_subset_features
    y = complete_subset_outcome

    dfs = []
    models = get_model()

    names = []
    for mods in models:
        names.append(mods.__class__.__name__)
        
    names = (', '.join(names))
    print(names)
        
    # results = []
    # scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    # target_names = ['Non-Asthma', 'Asthma']
    # classify_report = []

    sensitivity = list()
    specificity = list()
    PPV = list()
    NPV = list()
    LRp = list()
    LRn = list()
    ROCAUC_train = list()
    f1_score=list()
    precision_score = list()
    recall_score = list()
    accuracy_score = list()
    report = list()

    for model in models:

        
            y_pred = model.predict(X)
            cm_train = confusion_matrix(y, y_pred)
            #print(names)
            report = classification_report(y, y_pred, output_dict=True)
            #sccurate = accuracy_score(y_train, y_pred)
            #print(sccurate)
            
            macro_precision =  report['macro avg']['precision'] 
            macro_recall = report['macro avg']['recall']    
            macro_f1 = report['macro avg']['f1-score']
            
            accuracy = report['accuracy']
            print('jjjjj',macro_precision)
            #f1_score1 = (f1_score(y, y_pred))
            f1_score.append(macro_f1)
            
            #precision_score1 = (precision_score(y, y_pred))
            precision_score.append(macro_precision)
            
            #recall_score1 = (recall_score(y, y_pred)) 
            recall_score.append(macro_recall)
            
            #acc = accuracy_score(y, y_pred)
            accuracy_score.append(accuracy)
            
            sensitivity1 =  cm_train[1,1]/(cm_train[1,0]+cm_train[1,1])								
            #print(sensitivity)
            #0.6666666666666666
            sensitivity.append(sensitivity1)
            
            specificity1 = cm_train[0,0]/(cm_train[0,0]+cm_train[0,1])									
            #print(specificity)
            #0.89171974522293
            specificity.append( specificity1)
            
            
            PPV1 = cm_train[1,1]/(cm_train[1,1]+cm_train[0,1])	
            #print(PPV)
            #0.5
            PPV.append(PPV1)

            NPV1 = cm_train[0,0]/(cm_train[0,0]+cm_train[1,0])
            #print(NPV)
            #0.9427609427609428
            NPV.append( NPV1)

            LRp1 = sensitivity1/(1-specificity1)
            #print(LRp)
            #6.15686274509804
            LRp.append( LRp1)

            LRn1 = (1-sensitivity1)/specificity1
            #print(LRn)
            #0.3738095238095238
            LRn.append( LRn1)

            #  AUC: 
            probs = model.predict_proba(X)
            preds = probs[:,1]
            ROCAUC_train1 = roc_auc_score(y, preds)
            #print(ROCAUC_train)
            ROCAUC_train.append( ROCAUC_train1)
            


    print(names)
    models_initial = pd.DataFrame({
        
        'Model'       : (names),
        'sensitivity': [sensitivity],  
        'specificity': [specificity],
        'PPV'    : [PPV],
        'NPV'   : [NPV],
        'LRp'      : [LRp],
        'LRn'    : [LRn],
        'ROCAUC_train': [ROCAUC_train],
        'f1_score' : [f1_score],
        'precision_score' : [precision_score],
        'recall_score' : [recall_score],
        'accuracy_score' : [accuracy_score],
        }, columns = ['Model', 'ROCAUC_train', 'f1_score','precision_score', 'recall_score','accuracy_score' , 'sensitivity', 'specificity', 'PPV', 'NPV', 'LRp', 'LRn', 'ROCAUC_train'])


    #models_initial.sort_values(by='Accuracy', ascending=False)
    print(models_initial)
    print(models_initial.transpose())
    comapre = models_initial.transpose()
    comapre.to_csv(Output_name+'_FileComapring.csv', sep='\t', encoding='utf-8')
    #print("gdggdgdg", models.__class__.__name__)

    s3_resource.meta.client.upload_file( 
    Filename=Output_name+'_FileComapring.csv',Bucket='superlearner',Key=Output_name+'_FileComapring.csv')
   
    key= Output_name+'_FileComapring.csv'
    
    bucket = 'superlearner'
    New_url =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key}"
  

        # defining the api-endpoint  
    API_ENDPOINT = "https://file-ms-api.superlearnerscripts.com/process/complete_wone"
    payload = { "projectId": projectID,"processId": processID,"location": New_url, "name": key,
    "type": "csv"}
    headers = {'token': token}
    response = requests.post(API_ENDPOINT,  data = payload, headers= headers)


    #results.append(cv_results)
    #names.append(name)
    #this_df = pd.DataFrame(cv_results)
    #this_df['model'] = name
    #dfs.append(this_df)
    #final = pd.concat(dfs, ignore_index=True)
    #return final
    #print(final)
except Exception as ex:

    
    name = sys.argv[1]
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID, "projectId":projectID, "name":name, "reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)
    


    