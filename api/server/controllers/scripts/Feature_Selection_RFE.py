# Perform feature selection using Recursive Feature Elimination, with a 5-fold cross-validation

# Import packages
import os
import sys
import pandas as pd
import numpy as np
import imblearn
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn import preprocessing
#import json
#mport gridfs
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

	inFile = sys.argv[2] # file to perform feature_selection on
	File = pd.read_csv(inFile, index_col=False)

	# Remove individuals with missing data to create a subset of individuals with only complete data
	New_File = File.dropna()
	New_File.drop(New_File.filter(regex="Unname"),axis=1, inplace=True)
	# Separate the features (X) and outcome (Y) in paraparation for feature selection
	outc=sys.argv[4] #text box which is the outcome which is sys.arg[2] = Asthma_10YR, which should be entered in the textbox on the front end, that would display OUTCOME NAME 
	study=sys.argv[3] #sys.arg[3] = Study_ID, the study group which should be entered in the textbox on the frontend,  which will display STUDY/INDEX NAME
	filenamee=sys.argv[1] #new name for the file output

	study_1 = New_File[study]
	outc_1 = New_File[outc]
	
	outc_Y=New_File[[outc]]
	dataset_X=New_File.drop([outc] , axis='columns')
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
	
	#Define parameters for random forest algorithm to used for RFE (used default settings here) 
	best_param1= {'bootstrap': True,'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 100}

	# Used a balanced random forest classifier to account for the class imbalance in the dataset
	bclf = BalancedRandomForestClassifier(n_estimators=best_param1["n_estimators"],max_depth=best_param1["max_depth"],
								min_samples_split =best_param1["min_samples_split"],max_features=best_param1["max_features"],random_state=123)

	pal = list(ContinousVariable.columns)
	#### Define the RFE process ###
	# uses the balanced random forest classifer defined above, 
	# within a stratified 5-fold cross-validation (random states specfified to ensure the same splits each time, making it reproducible)
	# feature subset will be decided based on those that construct the model with the best 'score', in this case, the model performing with the best balanced accuracy (due to the class imbalance)
	rfecv = RFECV(estimator=bclf, step=1, cv=StratifiedKFold(5,random_state=123),
				scoring='balanced_accuracy')



	# Outline a pipeline to: standardise the continuous features > leave the categorical untouched > perform RFE			  
	estimators = Pipeline([
		('standardising', Pipeline([
			('select', ColumnTransformer([
				('scale', StandardScaler(), pal )
				],
				remainder='passthrough')
			)
		])),
	('bclf', rfecv)	
	])

	# Apply RFE to data

	Xs = Xs[(list(Xs.columns))]

	le = preprocessing.LabelEncoder()

	Xs[Catergorical.columns] = Xs[Catergorical.columns].apply(le.fit_transform)
	#X[Catergorical.columns] = le.fit_transform(list(Catergorical.columns))


	fit=estimators.fit(Xs, outc_Y.values.ravel())

	### Extract results ###
	# Label the features identified as belonging to the optimal subset of predictors
	list = []
	for i in range(0, 57):
		if rfecv.ranking_[i] == 1:
			list.append(Xs.columns.values[i])

	# Print the optimal number of features
	print("Optimal number of features : %d" % rfecv.n_features_)


	# Print the accuracy score that was obtained with the optimal number of features identified 
	print("Balanced Accuracy: \n", rfecv.grid_scores_[11]) # n features - 1 for indexing i.e. if the optimall subset included 12 features, 11 should be specified in this line of code

	# Print the list of features belonging to the optimal subset of predictors
	print("Feature Selected: \n",list)

	
	#outlier_removed = pd.concat([study_1,New_File,outc_1], axis=1)
	df = pd.DataFrame(New_File)
	df = df.loc[:, list]
	df = pd.concat([study_1,df,outc_1], axis=1)
	df.to_csv(filenamee+'_Feature_selected.csv')


	# Generate a plot for the number of features against their cross-validation scores
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross-validation balanced accuracy score")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.tight_layout()
	plt.savefig(filenamee+'_Feature_selection_graph.pdf')
	s3_resource.meta.client.upload_file(
		Filename=filenamee+'_Feature_selected.csv', Bucket='superlearner', Key=filenamee+'_Feature_selected.csv')

	s3_resource.meta.client.upload_file(
		Filename=filenamee+'_Feature_selection_graph.pdf', Bucket='superlearner', Key=filenamee+'_Feature_selection_graph.pdf')

	key= filenamee+'_Feature_selected.csv'
	key_two =filenamee+'_Feature_selection_graph.pdf'
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
    #key= filenamee+'_Feature_selected.csv'
    API_ENDPOINT_fail = "https://file-ms-api.superlearnerscripts.com/process/failed"
    payload = {"processId": processID, "projectId":projectID, "name":name, "reason":ex}
    headers = {'token': token}
    respnoses = requests.post(API_ENDPOINT_fail,  data = payload, headers= headers)