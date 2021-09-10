import make_null_predictions
import cal
import pandas as pd
import os
import sys
path = os.getcwd()
path = path.split('experiments')[0] + 'common'
# setting path for importing scripts
sys.path.insert(1, path)
import db_handler


# start pipeline, by intiating connection to database
# return connection as conn and cursor as cur
def start():
    print("Pipeline started for testing row 3 null hyposthesis experiment to predict in 4 hours.")
    conn = db_handler.intialize_database_handler()
    cur = conn.cursor()
    return conn, cur


# stop pipeline, by closing open connection and cursor to database
# return connection and cursor
def stop(conn, cur):
    db_handler.close_db_connection(conn, cur)
    print("Pipeline ended for testing row 3 null hyposthesis experiment to predict in 4 hours.")


if __name__ == "__main__":
    
    conn, cur = start()

    # read testing patients and run the pipeline for each 500 patients
    experiment = 'experiment_micu_testing.csv'

    pats_set = pd.read_csv(experiment)
    predictions = make_null_predictions.process_predict(conn, pats_set)

    print('calculating overall prediction results (precision, recall, F1-score) for testing row 3 null hyposthesis experiment to predict in 4 hours.')
    cal.calculate_results(conn, pats_set, predictions)

    stop(conn, cur)
    input('Press anything to continue....')