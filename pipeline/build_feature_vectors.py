import pandas as pd
import sys
import numpy as np
import multiprocessing
import threading
import time
import os
import pickle

# setting path for importing scripts in external folder
sys.path.insert(1, '../common')
sys.path.insert(3, '../task_3')
import db_handler
import build_diagnosis
import build_measurement


# Take feature vectors by patient and incooperate corrosponding demagrohic data in it
def enrich_demographic_features(adm_id, demo_data_adm_id):
    
    conn = db_handler.intialize_database_handler()

    pkl_file_demo = 'enrich_demo/fv_'
    
    features_vector_adm_id = pd.read_pickle(pkl_file_demo + str(adm_id) + '.pkl')
    val_pats = pd.read_csv('valid_admissions_age.csv')
    val_pats = val_pats[val_pats.hadm_id == adm_id]

    if len(val_pats) < 1:
        val_pats = pd.read_csv('pat_set.csv')
        val_pats = val_pats[val_pats.hadm_id == adm_id]

    features_vector_adm_id['age'] = val_pats['age'].iloc[0]
    features_vector_adm_id['ethnicity'] = demo_data_adm_id['ethnicity']
    features_vector_adm_id['gender'] = demo_data_adm_id['gender']
    features_vector_adm_id['insurance'] = demo_data_adm_id['insurance']

    features_vector_adm_id.to_pickle(pkl_file_demo + str(adm_id) + '.pkl')

    db_handler.close_db_connection(conn, conn.cursor())


# Take feature vectors by patient and incooperate corrosponding diagnosis data in it
def enrich_diagnosis_features(adm_id):
    conn = db_handler.intialize_database_handler()

    pkl_file_diag = 'enrich_diag/fv_'

    features_vector_adm_id = pd.read_pickle(pkl_file_diag + str(adm_id) + '.pkl')

    pat_diag_df = build_diagnosis.get_diagnosis_time_data_specific(
            conn, adm_id)

    for row in features_vector_adm_id.itertuples():
        t = getattr(row, 'time')

        pat_diag_df_tmp = pat_diag_df[pat_diag_df.timestamp <= t]
        if not pat_diag_df_tmp.empty:
            for j in range(0, 18):
                pat_diag_df_grp = pat_diag_df_tmp[pat_diag_df_tmp.higher_group == (
                    j + 1)]
                if len(pat_diag_df_grp) > 0:
                    features_vector_adm_id.at[row.Index, 'diagnosis_group_' + str(j + 1)] = 1

    features_vector_adm_id.to_pickle(pkl_file_diag + str(adm_id) + '.pkl')

    db_handler.close_db_connection(conn, conn.cursor())


# Take feature vectors by patient chunk and incooperate corrosponding measurement data in it
def process_meas_times(features_vector_adm_id, chunk_pkl_name, pat_meas_df):

    for row in features_vector_adm_id.itertuples():
        t = getattr(row, 'time')
        pat_meas_df_tmp = pat_meas_df[pat_meas_df.charttime <= t]

        if not pat_meas_df_tmp.empty:

            itm_grps = pat_meas_df_tmp.groupby('itemid').first().reset_index()
            cols_itms = list(map(lambda x:'meas_' + str(x), itm_grps['itemid'].tolist()))
            lst_meas = itm_grps['value'].tolist()

            features_vector_adm_id.at[row.Index,  cols_itms] = lst_meas

    features_vector_adm_id.to_pickle(chunk_pkl_name)

    del features_vector_adm_id


# Take feature vectors by patient chunk and incooperate corrosponding treatment data in it
def process_treat_times(features_vector_adm_id, chunk_pkl_name, pat_treat_df):

    all_treats_pats = pat_treat_df.mapped_id.unique()

    for row in features_vector_adm_id.itertuples():
        t = getattr(row, 'time')
    
        pat_treat_df_tmp = pat_treat_df[(pat_treat_df.starttime <= t)]

        trmts_grps = pat_treat_df_tmp.groupby('mapped_id').size().reset_index(name='counts')
        cols_trmts_gvn_times = list(map(lambda x:str(x) + '_given_times', trmts_grps['mapped_id'].tolist()))    

        count_trmts = trmts_grps['counts'].tolist()

        features_vector_adm_id.at[row.Index,  cols_trmts_gvn_times] = count_trmts

        pat_treat_df_tmp = pat_treat_df_tmp[pat_treat_df_tmp.endtime >= t]

        unique_trmts = []
        if not pat_treat_df_tmp.empty:
            unique_trmts = pat_treat_df_tmp.mapped_id.unique()
            
            cols_trmts_rcncy = list(map(lambda x: str(x) + '_recency' , unique_trmts))
            cols_trmts_gvn_nxt_4 = list(map(lambda x: str(x) + '_given_nxt_4' , unique_trmts))

            features_vector_adm_id.at[row.Index,  cols_trmts_rcncy] = 0
            features_vector_adm_id.at[row.Index,  cols_trmts_gvn_nxt_4] = 1
        
        tgn_4 = pd.to_datetime(t) + pd.DateOffset(hours=2)
        othr_trmts = list(set(all_treats_pats) - set(unique_trmts))

        pat_treat_df_tmp_trmts = pat_treat_df[pat_treat_df.mapped_id.isin(othr_trmts)]
        pat_treat_df_tmp = pat_treat_df_tmp_trmts[pat_treat_df_tmp_trmts.endtime < t]

        if not pat_treat_df_tmp.empty:

            trmts_grps = pat_treat_df_tmp.groupby('mapped_id').first().reset_index()
            cols_trmts_rcncy = list(map(lambda x:str(x) + '_recency', trmts_grps['mapped_id'].tolist()))

            rts = trmts_grps['endtime'].tolist()
            rmins = list(map(lambda x:round(((t-x)/np.timedelta64(1,'s'))/60), rts))

            features_vector_adm_id.at[row.Index,  cols_trmts_rcncy] = rmins

        pat_treat_df_tmp = pat_treat_df_tmp_trmts[pat_treat_df_tmp_trmts.starttime > t]
        pat_treat_df_tmp_4 = pat_treat_df_tmp[pat_treat_df_tmp.starttime <= tgn_4]

        if not pat_treat_df_tmp_4.empty:
            cols_trmts_gvn_nxt = list(map(lambda x: str(x) + '_given_nxt_4' , pat_treat_df_tmp_4.mapped_id.unique()))
            features_vector_adm_id.at[row.Index,  cols_trmts_gvn_nxt] = 1

    features_vector_adm_id.to_pickle(chunk_pkl_name)

    del features_vector_adm_id


# Take feature vectors by patient and made its chunks and call 
# corresponding calculation function
def chuk_features_vectors_spawn_cals(adm_id, pat_df, ftype):
    
    if ftype == 0:
        pkl_file = 'enrich_meas/fv_'
        log_stmt = 'Sub process function measuremnt ended'
        csize = 1000
        size = 1000
    else:
        pkl_file = 'enrich_treat/fv_'
        log_stmt = 'Sub process function treatment ended'
        csize = 1000
        size = 1000

    features_vector_adm_id = pd.read_pickle(pkl_file + str(adm_id) + '.pkl')
    

    if len(features_vector_adm_id) > size:
        list_of_dfs = [features_vector_adm_id.iloc[i:i+csize-1] for i in range(0, len(features_vector_adm_id),csize)]
        processes = []
        for i in range(0, len(list_of_dfs)):
            features_vector_adm_id_df = list_of_dfs[i]
            try:
                chunk_pkl_name = pkl_file + str(adm_id) + '_'+ str(i) + '.pkl'
                if not ftype:
                    p = multiprocessing.Process(target=process_meas_times, args=(features_vector_adm_id_df,chunk_pkl_name,pat_df,))
                else:
                    p = multiprocessing.Process(target=process_treat_times, args=(features_vector_adm_id_df,chunk_pkl_name,pat_df,))
                p.start()
            except:
                print("Error: unable to start Process")
                exit(0)
            processes.append(p)
        
        # Wait all process to finish.
        for p in processes:
            p.join()  
            p.terminate()

        features_vector_adm_id = pd.DataFrame()

        for i in range(0, len(list_of_dfs)):
            features_vector_adm_id_tmp = pd.read_pickle(pkl_file + str(adm_id) + '_'+ str(i) + '.pkl')
            features_vector_adm_id = features_vector_adm_id.append(features_vector_adm_id_tmp, ignore_index=True)

        features_vector_adm_id.to_pickle(pkl_file + str(adm_id) + '.pkl')
    else:
        if not ftype:
            process_meas_times(features_vector_adm_id, pkl_file + str(adm_id) + '.pkl', pat_df)
        else:
            process_treat_times(features_vector_adm_id, pkl_file + str(adm_id) + '.pkl', pat_df)


# return traning and testing data related to measurement, diagnosis and treatment vectors
def output_combine_features_vectors(unique_adm_ids, target_adm_id, items_num, items_cat):

    #Input different vectors types
    pkl_file_demo = 'enrich_demo/fv_'
    pkl_file_diag = 'enrich_diag/fv_'
    pkl_file_meas = 'enrich_meas/fv_'
    pkl_file_treat = 'enrich_treat/fv_'

    features_vectors_demo_diag_meas_train = pd.DataFrame()
    features_vectors_treat_train = pd.DataFrame()
    features_vectors_demo_diag_meas_test = pd.DataFrame()
    features_vectors_treat_test = pd.DataFrame()

    for adm_id in unique_adm_ids:

        df_demo = pd.read_pickle(pkl_file_demo + str(adm_id) + '.pkl')
        df_diag = pd.read_pickle(pkl_file_diag + str(adm_id) + '.pkl')
        df_meas = pd.read_pickle(pkl_file_meas + str(adm_id) + '.pkl')

        df_treat = pd.read_pickle(pkl_file_treat + str(adm_id) + '.pkl')

        df_tmp = pd.merge(df_demo, df_diag, on=['time','hadm_id'])
        df_tmp = pd.merge(df_tmp, df_meas,  on=['time','hadm_id'])

        features_vectors_demo_diag_meas_train = features_vectors_demo_diag_meas_train.append(df_tmp, ignore_index=True)
        features_vectors_treat_train = features_vectors_treat_train.append(df_treat, ignore_index=True)

    df_demo = pd.read_pickle(pkl_file_demo + str(target_adm_id) + '.pkl')
    df_diag = pd.read_pickle(pkl_file_diag + str(target_adm_id) + '.pkl')
    df_meas = pd.read_pickle(pkl_file_meas + str(target_adm_id) + '.pkl')

    df_treat = pd.read_pickle(pkl_file_treat + str(target_adm_id) + '.pkl')

    df_tmp = pd.merge(df_demo, df_diag, on=['time','hadm_id'])
    df_tmp = pd.merge(df_tmp, df_meas,  on=['time','hadm_id'])

    features_vectors_demo_diag_meas_test = features_vectors_demo_diag_meas_test.append(df_tmp, ignore_index=True)

    features_vectors_treat_test = features_vectors_treat_test.append(df_treat, ignore_index=True)

    features_vectors_demo_diag_meas = features_vectors_demo_diag_meas_train.append(features_vectors_demo_diag_meas_test, ignore_index=True)

    features_vectors_demo_diag_meas_train = features_vectors_demo_diag_meas[features_vectors_demo_diag_meas.hadm_id != target_adm_id]
    features_vectors_demo_diag_meas_test = features_vectors_demo_diag_meas[features_vectors_demo_diag_meas.hadm_id == target_adm_id]
    
    return features_vectors_demo_diag_meas_train,features_vectors_treat_train,features_vectors_demo_diag_meas_test,features_vectors_treat_test


#Start K threads to process demographic vector of each patient
def process_demogrphic_vectors(unique_adm_ids):

    print("in function process_demogrphic_vectors")

    conn = db_handler.intialize_database_handler()
    
    tmp_pats_query = '('
    for pat in unique_adm_ids:
        tmp_pats_query = tmp_pats_query + str(pat) + ', '
    tmp_pats_query = ", ".join(tmp_pats_query.split(", ")[0:-1])
    tmp_pats_query = tmp_pats_query + ')'

    demo_data_query = " SELECT ethnicity, gender, insurance, hadm_id FROM admissions "\
        "INNER JOIN d3sv1_patients_mv "\
        "on admissions.subject_id = d3sv1_patients_mv.subject_id "\
        "WHERE hadm_id IN " + tmp_pats_query + ';'
    demo_data = db_handler.make_selection_query(conn, demo_data_query)
    
    processes = []
    for adm_id in unique_adm_ids:

        demo_data_adm_id = demo_data[demo_data.hadm_id == adm_id].iloc[0]

        if not demo_data_adm_id.empty:
            try:
                p = multiprocessing.Process(target=enrich_demographic_features, args=(adm_id, demo_data_adm_id,))
                p.start()
            except:
                print("Error: unable to start Process")
                exit(0)
            processes.append(p)
    
    # Wait all processes to finish.
    for p in processes:
        p.join()
    
    db_handler.close_db_connection(conn, conn.cursor())

    print('Demographic vector calculated')


#Start K threads to process diagnosis vector of each patient
def process_diagnosis_vectors(unique_adm_ids):
    
    processes = []
    for adm_id in unique_adm_ids:
        try:
            p = multiprocessing.Process(target=enrich_diagnosis_features, args=(adm_id,))
            p.start()
            #print('Process Started')
        except:
            print("Error: unable to start thread")
            exit(0)
        processes.append(p)
    
    # Wait all process to finish.
    for p in processes:
        p.join()

    print('Diagnosis vector calculated')


#Start K processes to process measurement vector of each patient
def process_measurement_vectors(unique_adm_ids, items_num, items_cat):
    conn = db_handler.intialize_database_handler()

    items_num_lab = items_num[items_num.itemid < 220000]
    items_num_chart = items_num[items_num.itemid > 220000]
    items_cat_lab = items_cat[items_cat.itemid < 220000]
    items_cat_chart = items_cat[items_cat.itemid > 220000]

    all_pat_meas_num_chart_df = build_measurement.get_measurement_data_specific_items(
        conn, unique_adm_ids, items_num_chart.itemid, 0, 'd3sv1_chartevents_mv')
    all_pat_meas_num_lab_df = build_measurement.get_measurement_data_specific_items(
        conn, unique_adm_ids, items_num_lab.itemid, 0, 'd3sv1_labevents_mv')
    all_pat_meas_cat_chart_df = build_measurement.get_measurement_data_specific_items(
        conn, unique_adm_ids, items_cat_chart.itemid, 1, 'd3sv1_chartevents_mv')
    all_pat_meas_cat_lab_df = build_measurement.get_measurement_data_specific_items(
        conn, unique_adm_ids, items_cat_lab.itemid, 1, 'd3sv1_labevents_mv')

    
    all_pat_meas_num = all_pat_meas_num_chart_df.append(all_pat_meas_num_lab_df, ignore_index=True)
    all_pat_meas_cat = all_pat_meas_cat_chart_df.append(all_pat_meas_cat_lab_df, ignore_index=True)
    all_pat_meas = all_pat_meas_num.append(all_pat_meas_cat, ignore_index=True)
    all_pat_meas.sort_values(by=['charttime'], ascending=False,  inplace=True)

    chunks = [unique_adm_ids[x:x+100] for x in range(0, len(unique_adm_ids), 100)]
    #print(len(chunks))
    for adm_chunk in chunks:
        processes = []
        for adm_id in adm_chunk:
            try:
                all_pat_meas_mp = all_pat_meas[all_pat_meas.hadm_id == adm_id]
                p = multiprocessing.Process(target=chuk_features_vectors_spawn_cals, args=(adm_id, all_pat_meas_mp,0,))
                p.start()
                #print('Process Measurement Started')
            except:
                #print("Error: unable to start Process")
                exit(0)
            processes.append(p)
    
        # Wait all process chunks to finish.
        for p in processes:
            p.join()   
            p.terminate()

    db_handler.close_db_connection(conn, conn.cursor())
    print('measurement vector calculated')


#Start K processes to process treatment vector of each patient
def process_treatment_vectors(unique_adm_ids, treatments_df):
    treatments_df.sort_values(by=['endtime'], ascending=False,  inplace=True)

    chunks = [unique_adm_ids[x:x+100] for x in range(0, len(unique_adm_ids), 100)]

    for adm_chunk in chunks:
        processes = []
        for adm_id in adm_chunk:
            try:
                treatment_df_mp = treatments_df[treatments_df.hadm_id == adm_id]
                p = multiprocessing.Process(target=chuk_features_vectors_spawn_cals, args=(adm_id, treatment_df_mp,1,))
                p.start()
                #print('Process Treatment Started')
            except:
                #print("Error: unable to start Process")
                exit(0)
            processes.append(p)
        
        # Wait all process chunks to finish.
        for p in processes:
            p.join()
            p.terminate()

    print('treatement vector calculated')
