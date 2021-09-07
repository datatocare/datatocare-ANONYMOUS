## Pipeline Code Explanation
In this folder, you will find scripts and text files that will first make features vector and then using that make training data which will eventually be used to make models for predicting treatments to be given to input patients.

File listed in the folder are scripts: main.py, compute.py, evaluate.py, find_treatments.py, build_state_vectors.py, build_feature_vectors, build_models_predictions.py, helper.py and csv files: experiment_micu_testing.csv and valid_admissions_ever_micu_wo_holdout.csv.

Here is the description of each:

* **main.py** :
Main script file that handles connection to database and calls submodules to execute pipeline. 

* **compute.py**:
Script file that using discharge measurement tables computes statistics for numerical using quantiles, all values between the10𝑡ℎ quantile and the90𝑡ℎ quantile as normal; values below the 10𝑡ℎ quantile aslow abnormal and values above the 90𝑡ℎ quantile as high abnormal values

* **evaluate.py**:
Script file that using computed statistics first evaluate given patient state and then using patient state and statistics determine all-close patients.

* **find_treatments.py** :
Script file that find all treatments given to K/all-close patients and return them as Dataframe. 

* **build_base_vectors.py** :
Script file that first finds all times for which patients have some measurement taken or diagnosis made. Then base vectors are made by incorporating features vectors (demographics, diagnosis, measurements, treatments) with initial values. In last, vectors are partitioned by patients and features type. Partitions are created to enable concurrency through multiprocessing.

* **build_features_vectors.py** :
Script file that has functions that enrich features vectors. Each vectors type (Measurement, Treatment, Demographics, and Diagnosis) are calculated using separate processes for each patient.

* **build_models_predictions.py** :
Script file that process feature vectors to build prediction model for each treatment.

* **helper.py** :
Script file that get and return measurement type (categorical or numerical) information.

* **experiment_micu_testing.csv** :
A csv file that contains patients' information for testing. subject_id is the unique identifier that specifies an individual patient, hadm_id column is patient admission id and evaltime column is the time we evaluate the patient. admittime is the admission time of patient and timediff is the time difference in hours between admission and evaluation time. It also contains age since these patients belong to hold-out set, therefore the precomputed age information is not found in the valid_admissions_wo_holdout.csv file.

* **valid_admissions_wo_holdout.csv** :
A csv file that contains the patients who are ever admitted to the MICU. Additionally, it contains the age attribute of patients with respect to specific admission.

