import pandas as pd
import sys
# setting path for importing scripts in external folder
sys.path.insert(1, '../common')
import db_handler



# Read all measurements of a specific admission
# for specific itemids or measurements
#return measurements as dataframe
def get_measurement_data_specific_items(conn, unique_adm_ids, items, meas_type, table):
    meas_time_df_query = ''
    if not meas_type:
        meas_time_df_query = "SELECT hadm_id,itemid,valuenum as value,charttime, 0 as type FROM {0} "\
        "WHERE hadm_id in (".format(table)
    else:
        meas_time_df_query = "SELECT hadm_id,itemid,value,charttime, 1 as type FROM {0} "\
        "WHERE hadm_id in (".format(table)
    
    for adm_id in unique_adm_ids:
        meas_time_df_query = meas_time_df_query + str(adm_id) + ', '
    meas_time_df_query = ", ".join(meas_time_df_query.split(", ")[0:-1])
    meas_time_df_query = meas_time_df_query + ') '
    
    if not meas_type:
        meas_time_df_query = meas_time_df_query + 'and valuenum is not null and itemid in ('
    else:
        meas_time_df_query = meas_time_df_query + 'and valuenum is null and itemid in ('

    if not len(items):
        return pd.DataFrame()
    
    meas_time_df_query_itm = ''
    for itm in items:
        meas_time_df_query_itm = meas_time_df_query_itm + str(itm) + ', '
    meas_time_df_query_itm = ", ".join(meas_time_df_query_itm.split(", ")[0:-1])

    if meas_time_df_query_itm:
        meas_time_df_query = meas_time_df_query + meas_time_df_query_itm + ');'
        return db_handler.make_selection_query(conn, meas_time_df_query)
    else:
        return pd.DataFrame()


# Build measurement type table 
# required by Prediction pipleine in task_4
def build(cur):
    create_measurement_type(cur)
