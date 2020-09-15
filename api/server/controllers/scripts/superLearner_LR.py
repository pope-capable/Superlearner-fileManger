## example of a super learner model for binary classification
from numpy import hstack
from numpy import vstack
from numpy import asarray
import sys
import pickle 
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
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
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
import numpy as np
import boto3
import inline as inline
import requests 
import urllib.request
import json

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
#np.random.seed(123) 

projectID = sys.argv[5]
token =sys.argv[6]
processID = sys.argv[7]
#np.random.seed(123) 


try:
    # create a list of base-models
    inFile = sys.argv[2] # file to perform feature_selection on
    study=sys.argv[3] #text box which is the outcome which is sys.arg[2] = Asthma_10YR, which should be entered in the textbox on the front end, that would display OUTCOME NAME 
    outc=sys.argv[4] #sys.arg[3] = Study_ID, the study group which should be entered in the textbox on the frontend,  which will display STUDY/INDEX NAME
    Output_name = sys.argv[1]
    modelings = sys.argv[8]
    

    
    modelings = modelings.replace("[", "")
    modelings = modelings.replace("]", " ")
    modelings = modelings.replace("'", "")
    modelings = modelings.split(",")

   
    # create a list of base-models
    
    def get_model():
        model = list()
        for arg in modelings:
            model.append(pickle.load(urllib.request.urlopen(arg)))
        return model
        

    # fit all base models on the training dataset

    #Creating a Meta_X and Meta_y all base models on the training dataset
    tfile = open(Output_name+'test.txt', 'a')
    

    #Storing the model choosen to train the superlearner 

    

    #Creating a Meta_X and Meta_y all base models on the training dataset

    def get_out_of_fold_predictions(X, y, model):
        meta_X, meta_y = list(), list()
        # define split of data
        kfold = KFold(n_splits=10, shuffle=True)
        # enumerate splits
        for train_ix, test_ix in kfold.split(X):
            fold_yhats = list()
            # get data
            train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
            train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
            meta_y.extend(test_y)
            # fit and make predictions with each sub-model
            for mod in model:
                mods =mod.fit(train_X, train_y.astype(int))
                yhat = mods.predict_proba(test_X)
                
                # store columns
                fold_yhats.append(yhat)
            # store fold yhats as columns
            meta_X.append(hstack(fold_yhats))
        return vstack(meta_X), asarray(meta_y)

    file1 = open('Report.txt', 'w')
    sensitivity = list()
    sensitivity_test = list()
    specificity = list()
    specificity_test = list()
    PPV = list()
    PPV_test = list()
    NPV = list()
    NPV_test = list()
    LRp = list()
    LRp_test = list()
    LRn = list()
    LRn_test = list()
    ROCAUC_train = list()
    ROCAUC_test = list()
    f1_score=list()
    f1_score_test=list()
    precision_score = list()
    precision_score_test = list()
    recall_score = list()
    recall_score_test = list()
    accuracy_score1 = list()
    accuracy_score1_test = list()
    report = list()
    tfile = open(Output_name+'test.txt', 'a')


    def fit_base_models(X, y, models):
        modeling = list()
        for model in models:
            modelss= model.fit(X, y.astype(int))
            modeling.append(modelss)
        return modeling


    # evaluate a list of models on a dataset
    def evaluate_model(X, y, model):

        names = []
        for mods in model:
            names.append(mods.__class__.__name__)
        #print('sdsadcsdcsd',names)
        names = (', '.join(names))
        for mod in model:
        # mod.fit(X,y)
            yhat = mod.predict(X)
            acc = accuracy_score(y, yhat) 
            
            #ROCAUC = roc_auc_score(y, yhat) 
            probs = mod.predict_proba(X)
            preds = probs[:,1]
            ROCAUC = roc_auc_score(y, preds)
            ROCAUC_train.append("%.2f" % (ROCAUC * 100))

            #print(ROCAUC_train)
            print('%s: %.3f' % (mod.__class__.__name__, acc*100), "Accuracy on Training")
            print('%s: %.3f' % (mod.__class__.__name__, ROCAUC*100), "AUC on Training")
            cm_train = confusion_matrix(y, yhat)
            #print(names)
            report = classification_report(y, yhat, output_dict=True)
            
            macro_precision =  report['macro avg']['precision'] 
            macro_recall = report['macro avg']['recall']	
            macro_f1 = report['macro avg']['f1-score']
            
            accuracy = report['accuracy']
            
            f1_score.append("%.2f" % (macro_f1 * 100))
            
            #precision_score1 = (precision_score(y, yhat))
            precision_score.append("%.2f" % (macro_precision * 100))
            
            #recall_score1 = (recall_score(y, yhat)) 
            recall_score.append("%.2f" % (macro_recall * 100))
            
            #acc = accuracy_score(y, yhat)
            accuracy_score1.append("%.2f" % (accuracy * 100))
            
            sensitivity1 =	cm_train[1,1]/(cm_train[1,0]+cm_train[1,1])								
            #print(sensitivity)
            #0.6666666666666666
            sensitivity.append("%.2f" % (sensitivity1 * 100))
            
            specificity1 = cm_train[0,0]/(cm_train[0,0]+cm_train[0,1])									
            #print(specificity)
            #0.89171974522293
            specificity.append( "%.2f" % (specificity1 * 100))
            
            
            PPV1 = cm_train[1,1]/(cm_train[1,1]+cm_train[0,1])	
            #print(PPV)
            #0.5
            PPV.append("%.2f" % (PPV1 * 100) )

            NPV1 = cm_train[0,0]/(cm_train[0,0]+cm_train[1,0])
            #print(NPV)
            #0.9427609427609428
            NPV.append("%.2f" % ( NPV1 * 100))

            LRp1 = sensitivity1/(1-specificity1)
            #print(LRp)
            #6.15686274509804
            LRp.append("%.2f" % (LRp1 * 100))

            LRn1 = (1-sensitivity1)/specificity1
            #print(LRn)
            #0.3738095238095238
            LRn.append("%.2f" % (LRn1 * 100))
            

        models_initial = pd.DataFrame({
        
        'Model'		  : (names),
        'sensitivity': [sensitivity],  
        'specificity': [specificity],
        'PPV'	 : [PPV],
        'NPV'	: [NPV],
        'LRp'	   : [LRp],
        'LRn'	 : [LRn],
        'ROCAUC_train': [ROCAUC_train],
        'f1_score' : [f1_score],
        'precision_score' : [precision_score],
        'recall_score' : [recall_score],
        'accuracy_score' : [accuracy_score1],
        }, columns = ['Model', 'ROCAUC_train', 'f1_score','precision_score', 'recall_score','accuracy_score' , 'sensitivity', 'specificity', 'PPV', 'NPV', 'LRp', 'LRn'])
        comapre = models_initial.transpose()
        
        tfile.write('\n\n Training report of base report list below \n\n')
        tfile.write(comapre.to_string())

        #comapre.to_csv(r'pandas.txt', header=None, index=None, sep=' ', mode='a')
        #print(comapre)
        



    # def evaluate_models(X, y, models):
        # for model in models:
            # yhat = model.predict(X)
            # acc = accuracy_score(y, yhat)
            # print('%s: %.3f' % (model.__class__.__name__, acc*100))
    
            
    def evaluate_testmodel(X, y, model):

        names = []
        for mods in model:
            names.append(mods.__class__.__name__)
        
        names = (', '.join(names))
        for mod in model:
        
            yhat2 = mod.predict(X)
            acc = accuracy_score(y, yhat2)
            #accuracy_score1.append(acc)
            probs = mod.predict_proba(X)
            preds = probs[:,1]
            ROCAUC = roc_auc_score(y, preds)
            ROCAUC_test.append("%.2f" % (ROCAUC * 100))
            #print(ROCAUC_train)
            print('%s: %.3f' % (mod.__class__.__name__, acc*100), "test_Accuracy test data")
            print('%s: %.3f' % (mod.__class__.__name__, ROCAUC*100), "test_AUC test data")
            cm_train = confusion_matrix(y, yhat2)
            #print(names)
            report = classification_report(y, yhat2, output_dict=True)
            #sccurate = accuracy_score(y_train, yhat2)
            #print(sccurate)
            
            macro_precision =  report['macro avg']['precision'] 
            macro_recall = report['macro avg']['recall']	
            macro_f1 = report['macro avg']['f1-score']
            
            accuracy = report['accuracy']
            #print('jjjjj',macro_precision)
            #f1_score1 = (f1_score(y, yhat2))
            f1_score_test.append("%.2f" % (macro_f1 * 100))
            
            #precision_score1 = (precision_score(y, yhat2))
            precision_score_test.append("%.2f" % (macro_precision * 100))
            
            #recall_score1 = (recall_score(y, yhat2)) 
            recall_score_test.append("%.2f" % (macro_recall * 100))
            
            #acc = accuracy_score(y, yhat2)
            accuracy_score1_test.append("%.2f" % (accuracy * 100))
            
            sensitivity1 =	cm_train[1,1]/(cm_train[1,0]+cm_train[1,1])								
            
            sensitivity_test.append("%.2f" % (sensitivity1 * 100))
            
            specificity1 = cm_train[0,0]/(cm_train[0,0]+cm_train[0,1])									
            
            specificity_test.append("%.2f" % (specificity1 * 100))
            
            
            PPV1 = cm_train[1,1]/(cm_train[1,1]+cm_train[0,1])	
            
            PPV_test.append("%.2f" % (PPV1 * 100))

            NPV1 = cm_train[0,0]/(cm_train[0,0]+cm_train[1,0])
            
            NPV_test.append("%.2f" % (NPV1* 100))

            LRp1 = sensitivity1/(1-specificity1)

            LRp_test.append("%.2f" %(LRp1 * 100))

            LRn1 = (1-sensitivity1)/specificity1

            LRn_test.append("%.2f" % (LRn1 * 100))

            #  AUC: 
        

        models_initial1 = pd.DataFrame({
        
        'Model'		  : (names),
        'sensitivity': [sensitivity_test],	
        'specificity': [specificity_test],
        'PPV'	 : [PPV_test],
        'NPV'	: [NPV_test],
        'LRp'	   : [LRp_test],
        'LRn'	 : [LRn_test],
        'ROCAUC_test': [ROCAUC_test],
        'f1_score' : [f1_score_test],
        'precision_score' : [precision_score_test],
        'recall_score' : [recall_score_test],
        'accuracy_score' : [accuracy_score1_test],
        }, columns = ['Model', 'ROCAUC_test', 'f1_score','precision_score', 'recall_score','accuracy_score' , 'sensitivity', 'specificity', 'PPV', 'NPV', 'LRp', 'LRn'])
        
        coma = models_initial1.transpose()
        
        tfile.write('\n\n Test evaluation of base mode \n\n')
        tfile.write(coma.to_string())

        #comapre.to_csv(r'pandas.txt', header=None, index=None, sep=' ', mode='a')
        #print(comapre)
            

    
    # make predictions with stacked model
    def super_learner_predictions(X, model, meta_model):
        meta_X = list()
        for mod in model:
            yhat = mod.predict_proba(X)
            meta_X.append(yhat)
        meta_X = hstack(meta_X)
        return meta_model.predict(meta_X), ((meta_model.predict_proba(meta_X))[:, 1])
    
    # create the inputs and outputs

    ### Import and prepare data for model development ###
    # Import cleaned, unstandardised dataset containing all of the candidate predictors and outcome
    data_4YR = pd.read_csv(inFile, index_col=False)
    data_4YR = data_4YR.loc[:, ~data_4YR.columns.str.contains('^Unnamed')]

    # Subset the features selected from the feature selection method - RFE
    selected_data = data_4YR
    complete_data_4YR = selected_data.dropna()

    # Create data_features
    complete_subset_features =	complete_data_4YR.drop([study,outc], axis=1)

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
    #X, X_val, y, y_val = train_test_split(complete_subset_features,complete_subset_outcome)

    X_train, X_test, y_train, y_test = train_test_split(complete_subset_features, complete_subset_outcome,
                                                            stratify=complete_subset_outcome, 
                                                            test_size=0.40, shuffle = True, random_state=123)
        

    

    scaler = StandardScaler()
    # Standardise continuous features in the training set
    cont_train = pd.DataFrame(scaler.fit_transform(X_train.iloc[:,0:len(ContinousVariable)]), columns=(ContinousVariable))
    cat_train = X_train.iloc[:,len(ContinousVariable):]
    SX_train = pd.concat([cont_train, cat_train.reset_index(drop=True)], axis=1)

    cont_test = pd.DataFrame(scaler.transform(X_test.iloc[:,0:len(ContinousVariable)]), columns=(ContinousVariable))
    cat_test = X_test.iloc[:,len(ContinousVariable):]
    SX_test = pd.concat([cont_test, cat_test.reset_index(drop=True)], axis=1)

    model =get_model()
    meta_X, meta_y = get_out_of_fold_predictions(SX_train, y_train, model)
    print(meta_X)
    print(meta_y)

    
    modelings = fit_base_models(SX_train,y_train, model)
    with open(Output_name+'_Modelfile.mod', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(model, filehandle)
    print("shshs",modelings)
    
    meta_model = LogisticRegression(solver='liblinear', C=1.0, penalty = "l2", random_state=123).fit(meta_X, meta_y)

    
    meta_model.fit(meta_X, meta_y)

    filename = Output_name+'_SL_LR_finalized_model.mod'

    with open( filename, "wb") as file:
        pickle.dump(meta_model, file)

    
    tree_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=123).fit(meta_X, meta_y)

    filename = Output_name+'_SL_ETC_finalized_model.mod'
    with open( filename, "wb") as file:
        pickle.dump(tree_model, file)
    
    # fit the model on the whole dataset
    evaluate_model(SX_train, y_train, modelings)
    evaluate_testmodel(SX_test, y_test, modelings)


    yhat, preds1= super_learner_predictions(SX_test, modelings, meta_model)
    yhat2, preds2 = super_learner_predictions(SX_train, modelings, meta_model)
    #print(yhat)


    mhat, preds3 = super_learner_predictions(SX_test, model, tree_model)
    mhat2, preds4 = super_learner_predictions(SX_train, model, tree_model)

    ROCAUC1 = roc_auc_score(y_test, preds1)
    ROCAUC12 = roc_auc_score(y_train, preds2)

    ROCAUC2 = roc_auc_score(y_test, preds3)
    ROCAUC22 = roc_auc_score(y_train, preds4)


    #yhat = mod.predict(X)
    acc = accuracy_score(y_test, yhat)
    acc2 = accuracy_score(y_test, mhat)
    #accuracy_score1.append(acc)
    #probs = model.predict_proba(X)
    #preds = probs[:,1]
    #ROCAUC = roc_auc_score(y_test, yhat)
    #ROCAUCy = roc_auc_score(y_test, mhat)

    cm_train = confusion_matrix(y_train, yhat2)
    cm_train2 = confusion_matrix(y_train, mhat2)
    #print(names)
    report = classification_report(y_train, yhat2, output_dict=True)
    report2 = classification_report(y_train, mhat2, output_dict=True)
    #sccurate = accuracy_score(y_train, yhat)
    #print(sccurate)

    macro_precision =  report['macro avg']['precision'] 
    macro_precision2 =	report2['macro avg']['precision'] 
    macro_recall = report['macro avg']['recall']	
    macro_recall2 = report2['macro avg']['recall']	  
    macro_f1 = report['macro avg']['f1-score']
    macro_f12 = report2['macro avg']['f1-score']

    accuracy = report['accuracy']
    accuracy2 = report2['accuracy']

    sensitivity1 =	cm_train[1,1]/(cm_train[1,0]+cm_train[1,1])								
    sensitivity2 =	cm_train2[1,1]/(cm_train2[1,0]+cm_train2[1,1])							   

    specificity1 = cm_train[0,0]/(cm_train[0,0]+cm_train[0,1])									
    specificity2 = cm_train2[0,0]/(cm_train2[0,0]+cm_train2[0,1])								   

    PPV1 = cm_train[1,1]/(cm_train[1,1]+cm_train[0,1])	
    PPV2 = cm_train2[1,1]/(cm_train2[1,1]+cm_train2[0,1])  

    NPV1 = cm_train[0,0]/(cm_train[0,0]+cm_train[1,0])
    NPV2 = cm_train2[0,0]/(cm_train2[0,0]+cm_train2[1,0])

    LRp1 = sensitivity1/(1-specificity1)
    LRp2 = sensitivity2/(1-specificity2)

    LRn1 = (1-sensitivity1)/specificity1
    LRn2 = (1-sensitivity2)/specificity2

    models_initial = pd.DataFrame({

    'Model'		  : ('SuperLearner-LogisticRegression','SuperLearner-RandomForest' ),
    'sensitivity': [("%.2f" %(sensitivity1 * 100)),("%.2f" % (sensitivity2 * 100))],  
    'specificity': [("%.2f" % (specificity1 * 100)),("%.2f" % (specificity2 * 100))],
    'PPV'	 : [("%.2f" %(PPV1 * 100)),("%.2f" % (PPV2 * 100))],
    'NPV'	: [("%.2f" % (NPV1 * 100)),("%.2f" %(NPV2 * 100))],
    'LRp'	   : [("%.2f" % (LRp1 * 100)),("%.2f" %(LRp2*100))],
    'LRn'	 : [("%.2f" %(LRn1 * 100)),("%.2f" %(LRn2*100))],
    'ROCAUC_train': [("%.2f" %(ROCAUC12 * 100)),("%.2f" %(ROCAUC22 *	 100))],
    'f1_score' : [("%.2f" %(macro_f1 * 100)),("%.2f" %(macro_f12*100))],
    'precision_score' : [("%.2f" %(macro_precision* 100)),("%.2f" %(macro_precision2*100))],
    'recall_score' : [("%.2f" %(macro_recall * 100)),("%.2f" %(macro_recall2 * 100))],
    'accuracy_score' : [("%.2f" %(accuracy*100)),("%.2f" %(accuracy2*100))],
    }, columns = ['Model', 'ROCAUC_train', 'f1_score','precision_score', 'recall_score','accuracy_score' , 'sensitivity', 'specificity', 'PPV', 'NPV', 'LRp', 'LRn', 'ROCAUC_train'])
    comapre = models_initial.transpose()

    tfile.write("\n\n\n Train Evaluation For SuperLeaner Prediction \n\n\n")
    tfile.write(comapre.to_string())

    #yhat = mod.predict(X)
    acc = accuracy_score(y_test, yhat)
    acc2 = accuracy_score(y_test, mhat)
    #accuracy_score1.append(acc)
    #probs = model.predict_proba(X)
    #preds = probs[:,1]
    #ROCAUC = roc_auc_score(y_test, yhat)

    cm_train = confusion_matrix(y_test, yhat)
    cm_train2 = confusion_matrix(y_test, mhat)
    #print(names)
    report = classification_report(y_test, yhat, output_dict=True)
    report2 = classification_report(y_test, mhat, output_dict=True)
    #sccurate = accuracy_score(y_train, yhat)
    #print(sccurate)

    macro_precision =  report['macro avg']['precision'] 
    macro_precision2 =	report2['macro avg']['precision'] 
    macro_recall = report['macro avg']['recall']	
    macro_recall2 = report2['macro avg']['recall']	  
    macro_f1 = report['macro avg']['f1-score']
    macro_f12 = report2['macro avg']['f1-score']

    accuracy = report['accuracy']
    accuracy2 = report2['accuracy']

    sensitivity1 =	cm_train[1,1]/(cm_train[1,0]+cm_train[1,1])								
    sensitivity2 =	cm_train2[1,1]/(cm_train2[1,0]+cm_train2[1,1])							   

    specificity1 = cm_train[0,0]/(cm_train[0,0]+cm_train[0,1])									
    specificity2 = cm_train2[0,0]/(cm_train2[0,0]+cm_train2[0,1])								   

    PPV1 = cm_train[1,1]/(cm_train[1,1]+cm_train[0,1])	
    PPV2 = cm_train2[1,1]/(cm_train2[1,1]+cm_train2[0,1])  

    NPV1 = cm_train[0,0]/(cm_train[0,0]+cm_train[1,0])
    NPV2 = cm_train2[0,0]/(cm_train2[0,0]+cm_train2[1,0])

    LRp1 = sensitivity1/(1-specificity1)
    LRp2 = sensitivity2/(1-specificity2)

    LRn1 = (1-sensitivity1)/specificity1
    LRn2 = (1-sensitivity2)/specificity2

    models_initial = pd.DataFrame({

    'Model'		  : ('SuperLearner-LogisticRegression','SuperLearner-RandomForest' ),
    'sensitivity': [("%.2f" %(sensitivity1 * 100)),("%.2f" % (sensitivity2 * 100))],  
    'specificity': [("%.2f" % (specificity1 * 100)),("%.2f" % (specificity2 * 100))],
    'PPV'	 : [("%.2f" %(PPV1 * 100)),("%.2f" % (PPV2 * 100))],
    'NPV'	: [("%.2f" % (NPV1 * 100)),("%.2f" %(NPV2 * 100))],
    'LRp'	   : [("%.2f" % (LRp1 * 100)),("%.2f" %(LRp2*100))],
    'LRn'	 : [("%.2f" %(LRn1 * 100)),("%.2f" %(LRn2*100))],
    'ROCAUC': [("%.2f" %(ROCAUC1 * 100)),("%.2f" %(ROCAUC2 *  100))],
    'f1_score' : [("%.2f" %(macro_f1 * 100)),("%.2f" %(macro_f12*100))],
    'precision_score' : [("%.2f" %(macro_precision* 100)),("%.2f" %(macro_precision2*100))],
    'recall_score' : [("%.2f" %(macro_recall * 100)),("%.2f" %(macro_recall2 * 100))],
    'accuracy_score' : [("%.2f" %(acc*100)),("%.2f" %(acc2*100))],
    }, columns = ['Model', 'ROCAUC', 'f1_score','precision_score', 'recall_score','accuracy_score' , 'sensitivity', 'specificity', 'PPV', 'NPV', 'LRp', 'LRn'])
    comapre = models_initial.transpose()

    tfile.write("\n\n\n Test Evaluation For SuperLeaner Prediction \n\n\n")

    tfile.write(comapre.to_string())


    tfile.close()

    print('Super Learner Using Logistic Regresson: %.3f' % (accuracy_score(y_test, yhat) * 100), "Accuracy on test data")
    print('Super Learner Using Logistic Regresson: %.3f' % (ROCAUC1 * 100), "AUC on test data")
    print('Super Learner training Using Logistic Regresson: %.3f' % (accuracy_score(y_train, yhat2) * 100), "Accuracy on Training ")
    print('Super Learner train Using Logistic Regresson: %.3f' % (ROCAUC12 * 100), "AUC on Training")
    print('Super Learner using RandomForest Classifier: %.3f' % (accuracy_score(y_test, mhat) * 100), "Accuracy on test data")
    print('Super Learner using RandomForest Classifier: %.3f' % (ROCAUC2 * 100), "AUC on test data")
    print('Super Learner train using RandomForest Classifier: %.3f' % (accuracy_score(y_train, mhat2) * 100), "Accuracy on trainign data")

    print('Super Learner train using RandomForest Classifier: %.3f' % (ROCAUC22 * 100), "AUC on training data")

    s3_resource.meta.client.upload_file( 
        Filename=Output_name+'_Modelfile.mod',Bucket='superlearner',Key=Output_name+'_Modelfile.mod')
    s3_resource.meta.client.upload_file(
        Filename=Output_name+'_SL_LR_finalized_model.mod',Bucket='superlearner',Key=Output_name+'_SL_LR_finalized_model.mod')
    s3_resource.meta.client.upload_file(
        Filename=Output_name+'test.txt',Bucket='superlearner',Key=Output_name+'test.txt')
    
    key= Output_name+'_Modelfile.mod'
    key_two = Output_name+'_SL_LR_finalized_model.mod'
    key_three = Output_name+'test.txt'
    bucket = 'superlearner'
    New_url =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key}"
    New_url_two =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_two}"
    New_url_three =  f"https://{bucket}.s3.eu-west-2.amazonaws.com/{key_three}"



        # defining the api-endpoint  
    API_ENDPOINT = "https://file-ms-api.superlearnerscripts.com/super/create"
    payload = { "projectId": projectID,"processId": processID,"location1": New_url, "location2":New_url_two, "location3":New_url_three, "name3":key_three, "type3":"txt", "name": Output_name}
    headers = {'token': token}
    response = requests.post(API_ENDPOINT,  data = payload, headers= headers)

except Exception as ex:
    name = sys.argv[1]
    key= name
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID, "projectId":projectID, "name":key, "reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)
    print (ex)