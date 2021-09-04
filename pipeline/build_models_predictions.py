import multiprocessing
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
import random
from pandas.api.types import is_string_dtype
from sklearn.ensemble import RandomForestClassifier
import umap.umap_ as umap
from sklearn.utils import resample
import math
import numpy as np
import copy
import os
import psutil
import time
import gc
import sys
# setting path for importing scripts in external folder
sys.path.insert(1, '../common')
import db_handler


# Compile results to output actual and predicted treatment
def cal_prec_rec(hadm_id):
	rdf = pd.DataFrame()
	directory = 'results/' + str(hadm_id) + '/'

	for filename in os.listdir(directory):
		file_path = os.path.join(directory, filename)

		tdf = pd.read_pickle(file_path)
		rdf = rdf.append(tdf, ignore_index=True)

	df_pr = pd.DataFrame(columns=['time','state','precision','recall','treatments_predict','treatments_correctly_predicted','treatments_actual'])

	rdf = rdf[rdf.state == 0]
	states = rdf.state.unique()
	times = rdf.time.unique()

	for s in states:
		rdf_tmp = rdf[rdf.state == s]
		
		rdf_tmp_t = rdf_tmp[rdf_tmp.test == 1]
		ta = rdf_tmp_t.treatment.unique()
		rdf_tmp_p = rdf_tmp[rdf_tmp.predict == 1]
		tp = rdf_tmp_p.treatment.unique()
		print('************Actual Treatments***************')
		print(ta)
		print('********************************************')
		print('************Predicted Treatments************')
		print(tp)
		print('********************************************')
		p = 0
		r = 0

		if len(tp) > 0:
			p = len(set(tp) & set(ta))/len(tp)

		if len(ta) > 0:
			r = len(set(tp) & set(ta))/len(ta)

		if len(ta) > 0:
			df_pr_dict = {
			'time': times[s],
			'state' : s,
			'precision': p, 
			'recall': r,
			'treatments_predict': len(tp),
			'treatments_correctly_predicted': len(set(tp) & set(ta)),
			'treatments_actual': len(ta),
			}

			df_pr = df_pr.append(df_pr_dict, ignore_index=True)

	df_pr = df_pr.sort_values(by=['state', 'time'], ascending=True)
	df_pr.to_csv('results/' + str(hadm_id) +  '_results_accuracy_precision_recall.csv')


#build model basing upon randomforest and make predictions
def build_model(X_train,Y_train,X_test,Y_test,times,states,treat,hadm_id, nxt):

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

	nrdf.to_pickle('results/' + str(hadm_id) + '/'+ str(treat) + '_results_rf_' + str(nxt) + '.pkl')


# Build attributes and models by calling relevant function if memory is available 
def build_attributes_model_predict(ddmtr_df,ttr_df,ddmtte_df,cols_mddt,treat,times,states,hadm_id):
	while True:
		if psutil.virtual_memory().percent < 75:
			break
		else:
			time.sleep(60)
	print('Building attributes')
	X_train,Y_train_4,X_test,Y_test_4 = build_attributes_label(ddmtr_df,ttr_df,ddmtte_df,cols_mddt,treat)
	gc.collect()

	print('Building model')
	build_model(X_train,Y_train_4,X_test,Y_test_4,times,states,treat,hadm_id,4)
	gc.collect()


#Seperate attributes and labels from training and testing data
def build_attributes_label(ddmtr_df,ttr_df,ddmtte_df,cols_mddt,treat):

	cols = ['time', 'hadm_id', treat + '_given_times', treat + '_recency', treat + '_given_nxt_4']
	ttr_df_trt_df = ttr_df[cols]
	train = pd.merge(ddmtr_df, ttr_df_trt_df, on=['time','hadm_id'])
	train.drop(['time', 'hadm_id'], axis=1, inplace=True)
	Y_train_4 = train[treat + '_given_nxt_4']
	Y_train_4 = Y_train_4.astype('int')
	X_train = train[train.columns.difference([treat + '_given_nxt_4'])]
	X_train[treat + '_given_times'] = pd.to_numeric(X_train[treat + '_given_times'])
	X_train[treat + '_recency'] = pd.to_numeric(X_train[treat + '_recency'])

	Y_test_4 = ddmtte_df[treat + '_given_nxt_4']
	Y_test_4 = Y_test_4.astype('int')
	cols_x_test = copy.deepcopy(list(cols_mddt))
	cols_x_test.insert(0, treat + '_given_times')
	cols_x_test.insert(1, treat + '_recency')
	X_test = ddmtte_df[cols_x_test]
	X_test[treat + '_given_times'] = pd.to_numeric(X_test[treat + '_given_times'])
	X_test[treat + '_recency'] = pd.to_numeric(X_test[treat + '_recency'])

	return X_train,Y_train_4,X_test,Y_test_4


# Read all measurements type information by itemid
# from table d3sv1_measurement_items_type
# type 0 means numerical, 1 means categorical 
def get_measurements_type():
	conn = db_handler.intialize_database_handler()
	meas_type_df_query = "SELECT itemid, type FROM d3sv1_measurement_items_type;"
	meas_type_df = db_handler.make_selection_query(conn, meas_type_df_query)
	db_handler.close_db_connection(conn, conn.cursor())
	return meas_type_df


#Read training and testing data and transform categorical vectors to numerical vectors
def read_training_testing_dataframes(hadm_id,training_meas_diag_demo, training_treat, testing_meas_diag_demo, testing_treat):

	print('labelling data')

	# get abnormal percentiles for each measurement
	os.chdir("../task_1")
	df_num_cmpt = pd.read_pickle('numeric_computaion.pkl')
	os.chdir("../task_4")

	meas_types_df = get_measurements_type()
	meas_types_df['itemid'] = pd.to_numeric(meas_types_df['itemid'])
	num_meas = meas_types_df[meas_types_df.type == 0].itemid.unique().tolist()
	num_meas = ['meas_' + str(meas) for meas in num_meas]
	cat_meas = meas_types_df[meas_types_df.type == 1].itemid.unique().tolist()
	cat_meas = ['meas_' + str(meas) for meas in cat_meas]

	label_meas_diag_demo = training_meas_diag_demo.append(testing_meas_diag_demo, ignore_index=True)
	label_meas_diag_demo.reset_index(inplace=True)

	num_cols = label_meas_diag_demo.columns[label_meas_diag_demo.columns.isin(num_meas)]
	cat_cols = label_meas_diag_demo.columns[label_meas_diag_demo.columns.isin(cat_meas)]

	# custom label encoding of numerical measurements
	# 0 means value of numerical measurement is normal
	# 1 means values of numerical measurement is abnormal of type high
	# -1 means values of numerical measurement is abnormal of type low
	# Null values in measurement is encoded as 0 in measurement column, 
	# and one-hot encoded in a seperate column
	num_null_cols = []
	for col in num_cols:

		vals_dict = {}
		vals = []
		if label_meas_diag_demo[col].isnull().any():
			label_meas_diag_demo[col + '_null'] = 0
			num_null_cols.append(col + '_null')

			index = label_meas_diag_demo[col].index[label_meas_diag_demo[col].apply(pd.isnull)]
			
			label_meas_diag_demo.at[index,  col + '_null'] = 1

			vals_dict[np.nan] = 0

			vals = label_meas_diag_demo[col].dropna().unique().tolist()
		else:
			vals = label_meas_diag_demo[col].unique().tolist()

		meas_bounds = df_num_cmpt[df_num_cmpt.itemid == int(col.split('_')[-1])].iloc[0]

		for val in vals:
			if (val <= meas_bounds['up']) and (val >= meas_bounds['lp']):
				vals_dict[val] = 0
			elif (val > meas_bounds['up']):
				vals_dict[val] = 1               
			elif (val < meas_bounds['lp']):
				vals_dict[val] = -1

		label_meas_diag_demo[col].replace(vals_dict, inplace=True)


	#hot encoding categorical measurements and demographics
	demo_cols = ['gender','ethnicity','insurance']
	cat_demo_cols = copy.deepcopy(cat_cols.tolist())
	cat_demo_cols.extend(demo_cols)
	label_meas_diag_demo = pd.get_dummies(data=label_meas_diag_demo, columns=cat_demo_cols, dummy_na = True)

	#remove columns that are constant so doesn't add any information
	for col in label_meas_diag_demo.columns:
		if len(label_meas_diag_demo[col].unique()) == 1:
			label_meas_diag_demo.drop(col,inplace=True,axis=1)

	diag_cols = [x for x in label_meas_diag_demo.columns if 'diagnosis_group' in x]

	print('reducing data')

	#reducing cat cols
	non_cat_cols = ['time','hadm_id','index', 'age']
	non_cat_cols.extend(num_cols)
	non_cat_cols.extend(diag_cols)
	non_cat_cols.extend(num_null_cols)
	cat_cols = label_meas_diag_demo.columns.difference(non_cat_cols)

	dimensions = 5
	if len(cat_cols) < 100:
		dimensions = 2
	if len(cat_cols) > 1000:
		dimensions = 9
	new_cat_cols = []
	if len(cat_cols) > 0:
		print('reducing categorical features from ' + str(len(cat_cols)) + ' dimesions to ' + str(dimensions))
		umap_data = umap.UMAP(init='random',n_neighbors=15, min_dist=0.2, n_components=dimensions, n_epochs=200).fit_transform(label_meas_diag_demo[cat_cols].values)

		label_meas_diag_demo.drop(cat_cols, inplace=True,axis=1)
		
		for i in range(0,dimensions):
			new_cat_cols.append('umap_dim_cat_' + str(i))
			label_meas_diag_demo['umap_dim_cat_' + str(i)] = umap_data[:,i]

	#reducing diagnosis columns 
	new_diag_cols = []
	dimensions = 2
	if len(diag_cols) > 0:
		print('reducing diagnosis features from ' + str(len(diag_cols)) +  ' to 2')

		umap_data = umap.UMAP(init='random',n_neighbors=15, min_dist=0.2, n_components=dimensions, n_epochs=200).fit_transform(label_meas_diag_demo[diag_cols].values)
		
		label_meas_diag_demo.drop(diag_cols, inplace=True,axis=1)
		
		for i in range(0,dimensions):
			new_diag_cols.append('umap_dim_diag_' + str(i))
			label_meas_diag_demo['umap_dim_diag_' + str(i)] = umap_data[:,i]

	#reducing numerical columns 
	non_num_cols = ['time','hadm_id','index', 'age']
	non_num_cols.extend(new_cat_cols)
	non_num_cols.extend(new_diag_cols)
	all_num_cols = label_meas_diag_demo.columns.difference(non_num_cols)

	dimensions = 5
	if len(all_num_cols) < 100:
		dimensions = 2
	if len(all_num_cols) > 1000:
		dimensions = 9

	if len(all_num_cols) > 0:
		print('reducing numerical features from ' + str(len(all_num_cols)) + ' dimesions to ' + str(dimensions))
	
		umap_data = umap.UMAP(init='random',n_neighbors=15, min_dist=0.2, n_components=dimensions, n_epochs=200).fit_transform(label_meas_diag_demo[all_num_cols].values)
		
		label_meas_diag_demo.drop(all_num_cols, inplace=True,axis=1)

		for i in range(0,dimensions):
			label_meas_diag_demo['umap_dim_num_' + str(i)] = umap_data[:,i]

	training_meas_diag_demo = label_meas_diag_demo[label_meas_diag_demo.hadm_id != hadm_id]
	testing_meas_diag_demo = label_meas_diag_demo[label_meas_diag_demo.hadm_id == hadm_id]

	print('making testing data')

	cols_mddt = testing_meas_diag_demo.columns.difference(['time', 'hadm_id', 'index'])
	cols_tt = testing_treat.columns.difference(['hadm_id', 'index'])
	cols_t = copy.deepcopy(list(cols_mddt))
	cols_t.extend(list(cols_tt))
	cols_t.append('state')

	testing_data = pd.DataFrame(columns=cols_t)
	times = testing_meas_diag_demo['time']

	del testing_meas_diag_demo['index']

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
	print('build called')
	ddmtr_df,ttr_df,ddmtte_df,cols_mddt = read_training_testing_dataframes(hadm_id,training_meas_diag_demo, training_treat, testing_meas_diag_demo, testing_treat)

	times = ddmtte_df['time']
	states = ddmtte_df['state']
	del ddmtr_df['index']

	treatments = []

	for col in ttr_df.columns:
		if ('_given_times' in col):
			treatments.append(col.split('_')[0])

	print('Total Treatments')
	print(len(treatments))

	directory = 'results/' +  str(hadm_id)
	if not os.path.exists(directory):
		os.makedirs(directory)

	print('Building model and predicting treatments')

	chunk_size =  int(len(treatments)/3) + 1 
	chunks = [treatments[x:x+chunk_size] for x in range(0, len(treatments), chunk_size)]
	count = 0
	for treat_chunk in chunks:
		processes = []
		print('Processing Chunk {0}'.format(count))
		for treat in treat_chunk:
			print(treat)
			try:
				p = multiprocessing.Process(target=build_attributes_model_predict, args=(ddmtr_df,ttr_df,ddmtte_df,cols_mddt,str(treat),times,states,hadm_id,))
				p.start()
			except:
				exit(0)
			processes.append(p)

		# Wait all process chunks to finish.
		for p in processes:
			p.join()
			p.terminate()
			gc.collect()

		print('Chunk {0} Processed'.format(count))
		count += 1

	print('Treatement predicted')
