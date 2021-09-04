import pandas as pd
import sys
import os
import copy
import numpy as np

# setting path for importing scripts in external folder
sys.path.insert(1, '../common')
sys.path.insert(3, '../task_3')
import db_handler


def intialize_base_enrich_dirs():
    dircs = ['enrich_demo/','enrich_diag/','enrich_meas/','enrich_treat/']

    for dirc in dircs:
        for filename in os.listdir(dirc):
            file_path = os.path.join(dirc, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


#Partition base vectors by patients and feature type for concurency
def partition_base_vector(base_vectors, demo_cols, diag_cols, meas_cols, treat_cols):
    unique_adm_ids = base_vectors.hadm_id.unique()
    pkl_file_demo = 'enrich_demo/fv_'
    pkl_file_diag = 'enrich_diag/fv_'
    pkl_file_meas = 'enrich_meas/fv_'
    pkl_file_treat = 'enrich_treat/fv_'
    
    for adm_id in unique_adm_ids:
        base_vectors_adm_id = base_vectors[base_vectors.hadm_id == adm_id]
        
        base_vectors_adm_id_tmp = base_vectors_adm_id[demo_cols]
        base_vectors_adm_id_tmp.to_pickle(pkl_file_demo + str(adm_id) + '.pkl')
        
        base_vectors_adm_id_tmp = base_vectors_adm_id[diag_cols]
        base_vectors_adm_id_tmp.to_pickle(pkl_file_diag + str(adm_id) + '.pkl')
        
        base_vectors_adm_id_tmp = base_vectors_adm_id[meas_cols]
        base_vectors_adm_id_tmp.to_pickle(pkl_file_meas + str(adm_id) + '.pkl')
        
        base_vectors_adm_id_tmp = base_vectors_adm_id[treat_cols]
        base_vectors_adm_id_tmp.to_pickle(pkl_file_treat + str(adm_id) + '.pkl')


# Create and intialize features and treatment vectors columns
def intialize_base_vectors(times_df, meas_itms_vec_num, meas_itms_vec_cat, treatments):
    
    base_cols = ['hadm_id','time']
    base_vectors = times_df
    
    base_vectors['age'] = [0] * len(times_df)
    base_vectors['gender'] = ['F'] * len(times_df)
    base_vectors['ethnicity'] = ['WHITE'] * len(times_df)
    base_vectors['insurance'] = ['Private'] * len(times_df)
    demo_cols = copy.deepcopy(base_cols)
    demo_cols.append('age')
    demo_cols.append('gender')
    demo_cols.append('ethnicity')
    demo_cols.append('insurance')
    
    diag_cols = copy.deepcopy(base_cols)
    for i in range(0, 18):
        col = 'diagnosis_group_' + str(i + 1)
        diag_cols.append(col)
        base_vectors[col] = [0] * len(times_df)
    
    meas_cols = copy.deepcopy(base_cols)
    for item in meas_itms_vec_num['itemid']:
        col = 'meas_' + str(item)
        meas_cols.append(col)
        base_vectors[col] = [np.nan] * len(times_df)

    for item in meas_itms_vec_cat['itemid']:
        col = 'meas_' + str(item)
        meas_cols.append(col)
        base_vectors[col] = [None] * len(times_df)
    
    treat_cols = copy.deepcopy(base_cols)
    for trmnt in treatments:
        col_rec = str(trmnt) + '_recency'
        col_tms = str(trmnt) + '_given_times'
        col_nxt_4 = str(trmnt) + '_given_nxt_4'
        treat_cols.append(col_rec)
        treat_cols.append(col_tms)
        treat_cols.append(col_nxt_4)
        base_vectors[col_rec] = [-1] * len(times_df)
        base_vectors[col_tms] = [0] * len(times_df)
        base_vectors[col_nxt_4] = [0] * len(times_df)
    
    partition_base_vector(base_vectors, demo_cols, diag_cols, meas_cols, treat_cols)



# Get items which will be used as measuremnts vector
# Return numrical and categorical measures seperately
def get_meas_items_features(conn, hadm_id, t, similar_patients):
    tmp_pats_query = '('
    for pat in similar_patients:
        tmp_pats_query = tmp_pats_query + str(pat) + ', '
    tmp_pats_query = ", ".join(tmp_pats_query.split(", ")[0:-1])
    tmp_pats_query = tmp_pats_query + ')'

    meas_itms_vec_num_chart_query = "SELECT DISTINCT itemid FROM d3sv1_chartevents_mv "\
        "WHERE valuenum IS NOT null and hadm_id IN " + tmp_pats_query + " INTERSECT "\
        "SELECT DISTINCT itemid FROM d3sv1_chartevents_mv "\
        "WHERE valuenum IS NOT null and hadm_id = {0} and charttime <= \'{1}\'; ".format(hadm_id, t)
    
    meas_itms_vec_cat_chart_query = "SELECT DISTINCT itemid FROM d3sv1_chartevents_mv "\
        "WHERE valuenum IS null and hadm_id IN " + tmp_pats_query + " INTERSECT "\
        "SELECT DISTINCT itemid FROM d3sv1_chartevents_mv "\
        "WHERE valuenum IS null and hadm_id = {0} and charttime <= \'{1}\'; ".format(hadm_id, t)

    meas_itms_vec_num_lab_query = "SELECT DISTINCT itemid FROM d3sv1_labevents_mv "\
        "WHERE valuenum IS NOT null and hadm_id IN " + tmp_pats_query + " INTERSECT "\
        "SELECT DISTINCT itemid FROM d3sv1_labevents_mv "\
        "WHERE valuenum IS NOT null and hadm_id = {0} and charttime <= \'{1}\'; ".format(hadm_id, t)
    
    meas_itms_vec_cat_lab_query = "SELECT DISTINCT itemid FROM d3sv1_labevents_mv "\
        "WHERE valuenum IS null and hadm_id IN " + tmp_pats_query + " INTERSECT "\
        "SELECT DISTINCT itemid FROM d3sv1_labevents_mv "\
        "WHERE valuenum IS null and hadm_id = {0} and charttime <= \'{1}\'; ".format(hadm_id, t)

    meas_itms_vec_num_chart = db_handler.make_selection_query(
        conn, meas_itms_vec_num_chart_query)
    meas_itms_vec_cat_chart = db_handler.make_selection_query(
        conn, meas_itms_vec_cat_chart_query)
    meas_itms_vec_num_lab = db_handler.make_selection_query(
        conn, meas_itms_vec_num_lab_query)
    meas_itms_vec_cat_lab = db_handler.make_selection_query(
        conn, meas_itms_vec_cat_lab_query)

    return meas_itms_vec_num_chart.append(meas_itms_vec_num_lab, ignore_index=True), meas_itms_vec_cat_chart.append(meas_itms_vec_cat_lab, ignore_index=True)


# Get all times for patients at which diagnosis or measurement was taken
# Return times for patients as a dataframe
def get_all_times(conn, hadm_id, t, similar_patients):
    tmp_similar_patients = copy.deepcopy(similar_patients)
    tmp_similar_patients.append(hadm_id)
    tmp_pats_query = '('
    for pat in tmp_similar_patients:
        tmp_pats_query = tmp_pats_query + str(pat) + ', '
    tmp_pats_query = ", ".join(tmp_pats_query.split(", ")[0:-1])

    tmp_pats_query = tmp_pats_query + ')'
    all_time_query = "SELECT hadm_id,charttime AS time FROM d3sv1_chartevents_mv "\
        "WHERE hadm_id IN " + tmp_pats_query + " UNION "\
        "SELECT hadm_id,timestamp AS time FROM d3sv1_patient_diagnosis_time "\
        "WHERE hadm_id IN " + tmp_pats_query + " UNION "\
        "SELECT hadm_id,charttime AS time FROM d3sv1_labevents_mv "\
        "WHERE hadm_id IN " + tmp_pats_query + ";"

    all_time_df = db_handler.make_selection_query(conn, all_time_query)
    time_pat_given_df = all_time_df[all_time_df.hadm_id == hadm_id]
    all_time_df = all_time_df[all_time_df.hadm_id != hadm_id]

    time_pat_given_df.sort_values(by=['time'], inplace=True, ascending=False)
    max_time = time_pat_given_df['time'].iloc[0]
    times_pat_given = []
    t = pd.to_datetime(t)
    while t <= max_time:
        times_pat_given.append(t)
        t = pd.to_datetime(t) + pd.DateOffset(hours=4)

    for time in times_pat_given:
        time_dict_pat_tmp = {
            'hadm_id': hadm_id,
            'time': time
        }
        all_time_df = all_time_df.append(time_dict_pat_tmp, ignore_index=True)
    return all_time_df


# Build base vectors by calling all required functions
def build(conn, cur, hadm_id, t, similar_patients, treatments_df):
    intialize_base_enrich_dirs()
    times_df = get_all_times(conn, hadm_id, t, similar_patients)
    meas_itms_vec_num, meas_itms_vec_cat = get_meas_items_features(
        conn, hadm_id, t, similar_patients)
    intialize_base_vectors(
        times_df, meas_itms_vec_num, meas_itms_vec_cat, treatments_df.mapped_id.unique())
    return meas_itms_vec_num, meas_itms_vec_cat
