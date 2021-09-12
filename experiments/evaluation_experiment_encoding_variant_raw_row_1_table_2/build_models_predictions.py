import multiprocessing
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
from sklearn.ensemble import RandomForestClassifier
import umap.umap_ as umap
import numpy as np
import copy
import psutil
import gc
import helper
import time
import lzma
import os
import sys
path = os.getcwd()
path = path.split('experiments')[0] + 'common'
# setting path for importing scripts
sys.path.insert(1, path)
import db_handler


# Compile results to output actual and predicted treatment
def cal_potential_results(hadm_id):
	rdf = pd.DataFrame()
	directory = 'results_treat_predict/' + str(hadm_id) + '/'

	for filename in os.listdir(directory):

		if 'rf_2' in filename:
			file_path = os.path.join(directory, filename)

			tdf = pd.read_pickle(file_path)
			rdf = rdf.append(tdf, ignore_index=True)

	rdf = rdf[rdf.state == 0]

	states = rdf.state.unique()
	times = rdf.time.unique()

	for s in states:
		rdf_tmp = rdf[rdf.state == s]
		

		rdf_tmp_t = rdf_tmp[rdf_tmp.test == 1]
		ta = rdf_tmp_t.treatment.unique()
		rdf_tmp_p = rdf_tmp[rdf_tmp.predict == 1]
		tp = rdf_tmp_p.treatment.unique()

		print('************Potential Actual Treatments in next 2 hours***************')
		print(ta)
		print('********************************************')
		print('************Predicted Treatments in next 2 hours************')
		print(tp)
		print('********************************************')


#build model basing upon randomforest and make predictions
def build_model(X_train,Y_train,X_test,Y_test,times,states,treat,hadm_id,version):

	clf=RandomForestClassifier(random_state=0)
	clf.fit(X_train,Y_train)
	Y_pred_rf =clf.predict(X_test)
	score = clf.score(X_test, Y_test)

	nrdf = pd.DataFrame()
	nrdf['time'] = times
	nrdf['state'] = states
	
	nrdf['treatment'] = [treat] * len(times)
	nrdf['test'] = Y_test.tolist()
	nrdf['predict'] = Y_pred_rf
	nrdf['score'] = score
	nrdf.to_pickle('results_treat_predict/' + str(hadm_id) + '/'+ str(treat) + '_results_rf_'+ str(version) + '.pkl')


# Build attributes and models by calling relevant function if memory is available 
def build_attributes_model_predict(ddmtr_df,ttr_df,ddmtte_df,cols_mddt,treat,times,states,hadm_id):

	print('Building attributes for treatment : ' + str(treat))
	X_train,Y_train_2,X_test,Y_test_2 = build_attributes_label(ddmtr_df,ttr_df,ddmtte_df,cols_mddt,treat)

	print('Building model for treatment : ' + str(treat))
	build_model(X_train,Y_train_2,X_test,Y_test_2,times,states,treat,hadm_id,2)


#Seperate attributes and labels from training and testing data for each potential treatment
def build_attributes_label(ddmtr_df,ttr_df,ddmtte_df,cols_mddt,treat):

	cols = ['time', 'hadm_id', treat + '_given_times', treat + '_recency', treat + '_given_nxt']

	ttr_df_trt_df = ttr_df[cols]
	train = pd.merge(ddmtr_df, ttr_df_trt_df, on=['time','hadm_id'])
	train.drop(['time', 'hadm_id'], axis=1, inplace=True)
	Y_train = train[treat + '_given_nxt']
	Y_train = Y_train.astype('int')

	X_train = train[train.columns.difference([treat + '_given_nxt'])]
	X_train[treat + '_given_times'] = pd.to_numeric(X_train[treat + '_given_times'])
	X_train[treat + '_recency'] = pd.to_numeric(X_train[treat + '_recency'])

	Y_test = ddmtte_df[treat + '_given_nxt']
	Y_test = Y_test.astype('int')

	cols_x_test = copy.deepcopy(list(cols_mddt))
	cols_x_test.insert(0, treat + '_given_times')
	cols_x_test.insert(1, treat + '_recency')
	X_test = ddmtte_df[cols_x_test]
	X_test[treat + '_given_times'] = pd.to_numeric(X_test[treat + '_given_times'])
	X_test[treat + '_recency'] = pd.to_numeric(X_test[treat + '_recency'])

	return X_train,Y_train,X_test,Y_test


#Make training and testing data,transform categorical vectors to numerical vectors and dimensionally reduced them
def build_training_testing_dataframes(hadm_id,training_meas_diag_demo, training_treat, testing_meas_diag_demo, testing_treat):

	print('seperating evaluating state data')

	cols_mddt = testing_meas_diag_demo.columns.difference(['time', 'hadm_id', 'index'])
	cols_tt = testing_treat.columns.difference(['hadm_id', 'index'])
	cols_t = copy.deepcopy(list(cols_mddt))
	cols_t.extend(list(cols_tt))
	cols_t.append('state')

	testing_data = pd.DataFrame(columns=cols_t)
	times = testing_meas_diag_demo['time']

	# del testing_meas_diag_demo['index']

	si = 0
	for t in times:
		mddt_df = testing_meas_diag_demo[testing_meas_diag_demo.time == t]
		tt_df = testing_treat[testing_treat.time == t]
		del mddt_df['time']
		tmp_df = pd.merge(mddt_df, tt_df, on=['hadm_id'])
		tmp_df = tmp_df.drop('hadm_id', axis=1)
		tmp_df['state'] = si
		si =si + 1
		testing_data = testing_data.append(tmp_df, ignore_index=True)

	return training_meas_diag_demo,training_treat,testing_data,cols_mddt


#Call functions to build training and testing attributes and labels and then using them 
#build model basing upon randomforest and make predictions
def build(hadm_id,training_meas_diag_demo, training_treat, testing_meas_diag_demo, testing_treat):
	
	print('build training/testing data, preiction models called')

	# directory that will contain each testing patient prediction results
	result_directory = 'results_treat_predict' 
	if not os.path.exists(result_directory):
		os.makedirs(result_directory)

	ddmtr_df,ttr_df,ddmtte_df,cols_mddt = build_training_testing_dataframes(hadm_id,training_meas_diag_demo, training_treat, testing_meas_diag_demo, testing_treat)

	times = ddmtte_df['time']
	states = ddmtte_df['state']
	
	treatments = []

	for col in ttr_df.columns:
		if ('_given_times' in col):
			treatments.append(col.split('_')[0])

	print('Total Treatments for prediction : ' + str(len(treatments)))

	directory = result_directory + '/' +  str(hadm_id)
	if not os.path.exists(directory):
		os.makedirs(directory)

	print('Building models and predicting treatments')

	chunk_size =  int(len(treatments)/4) + 1 
	chunks = [treatments[x:x+chunk_size] for x in range(0, len(treatments), chunk_size)]
	count = 0
	for treat_chunk in chunks:
		processes = []
		print('Processing Chunk {0}'.format(count))
		for treat in treat_chunk:
			try:
				p = multiprocessing.Process(target=build_attributes_model_predict, args=(ddmtr_df,ttr_df,ddmtte_df,cols_mddt,str(treat),times,states,hadm_id,))
				p.start()
			except:
				print('exception occured')
				exit(0)
			processes.append(p)

		# Wait all process chunks to finish.
		for p in processes:
			p.join()
			p.close()
			gc.collect()

		print('Chunk {0} Processed'.format(count))
		count += 1

	print('Treatements predicted')