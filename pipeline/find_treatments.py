import sys
import pickle
import os
import pandas as pd
# setting path for importing scripts in external folder
sys.path.insert(1, '../common')
sys.path.insert(2, '../task_1')
import db_handler
import evaluate


# read similar patients hadm_ids from pickel created by evaluate of task 1
def read_similar_patients(hadm_id):
    os.chdir("../task_1")
    similar_patients_pkl = 'results/' + str(hadm_id) + "_similar_patients.pkl"
    similar_patients_df = pd.DataFrame(columns = ['hadm_id','score'])

    with open(similar_patients_pkl, "rb") as f:
        similar_patients = pickle.load(f)
        for pat in similar_patients:
            tmp_pat_dict = {
                'hadm_id' : pat['hadm_id'],
                'score' : pat['score']
            }
            similar_patients_df = similar_patients_df.append(
                        tmp_pat_dict, ignore_index=True)

    similar_patients_df = similar_patients_df[similar_patients_df.score >= 70]
    os.chdir("../task_4")
    val_pats = pd.read_csv('valid_admissions_age.csv')
    valid_hadmids = val_pats.hadm_id.tolist()
    similar_patients_df['hadm_id'] = similar_patients_df['hadm_id'].astype(int)
    similar_patients_df = similar_patients_df[similar_patients_df['hadm_id'].isin(valid_hadmids)]

    if len(similar_patients_df) >= 200:
        similar_patients_df = similar_patients_df.sort_values(by=['score'], ascending=False)
        similar_patients_df = similar_patients_df.head(200)

    print(len(similar_patients_df))
    similar_patients = similar_patients_df['hadm_id'].tolist()
    return similar_patients


# Find all treatments given to K/all-close patients
# Return treatments as a dataframe
def get_all_treatments(conn, hadm_id, similar_patients):
    all_treat_query = "SELECT t1.hadm_id,t1.starttime,t1.endtime,t1.itemid,t2.label,t2.mapped_id "\
        "FROM inputevents_mv t1 INNER JOIN d3sv1_drugs_mapping t2 ON t1.itemid = t2.itemid "\
        "WHERE t2.mapping_level = 1 and t1.hadm_id IN ("
    for pat in similar_patients:
        all_treat_query = all_treat_query + str(pat) + ', '
    all_treat_query = ", ".join(all_treat_query.split(", ")[0:-1])
    all_treat_query = all_treat_query + ');'
    all_treat_df = db_handler.make_selection_query(conn, all_treat_query)

    mapped_ids = list(all_treat_df.mapped_id.unique())
    target_pat_query = "SELECT t1.hadm_id,t1.starttime,t1.endtime,t1.itemid,t2.label,t2.mapped_id "\
        "FROM inputevents_mv t1 INNER JOIN d3sv1_drugs_mapping t2 ON t1.itemid = t2.itemid "\
        "WHERE t1.hadm_id = " + str(hadm_id) + " and t2.mapped_id IN ("
    for mpid in mapped_ids:
        target_pat_query =  target_pat_query + str(mpid) + ', '
    target_pat_query = ", ".join(target_pat_query.split(", ")[0:-1])
    target_pat_query = target_pat_query + ');'
    target_treat_df = db_handler.make_selection_query(conn, target_pat_query)
    
    all_treat_df = all_treat_df.append(target_treat_df, ignore_index=True)
    return all_treat_df


# Build diagnosis vector by calling all required functions
def find(conn, cur, hadm_id,t):
    evaluate.evaluate(conn, hadm_id,t)
    similar_patients = read_similar_patients(hadm_id)
    if len(similar_patients) > 0:
        return similar_patients,get_all_treatments(conn, hadm_id, similar_patients)
    else:
        return [], pd.DataFrame()

