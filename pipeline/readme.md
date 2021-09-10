## Experiment Explanation
In this folder, you will find sub-folders that contains code to run a different flavor of pipeline corresponding to evaluation and testing experiments mentioned in table 2 and 3 of the paper.  

Here is a brief description of each experiment, more detailed explanation could be found in the paper:  
1. * **evaluation_experiment_encoding_variant_raw_row_1_table_2** :  
In this evaluation experiment, we evaluate the effect (prediction performance) of using raw encoding variant of feature vectors in the pipeline.  
2. * **evaluation_experiment_encoding_variant_abnormality-hot encoding_row_2_table_2** :  
In this evaluation experiment, we evaluate the effect of using normality-hot encoding variant of feature vectors.    
3. * **evaluation_experiment_encoding_variant_dimensionality reduction_row_3_table_2** :  
In this evaluation experiment, we evaluate the effect of using dimensionally reduced variant of feature vectors.  
4. * **evaluation_experiment_time_horizon_2-hours_row_4_table_2** :  
In this evaluation experiment, we evaluate the effect of using time horizon of 2 hours as treatment recommendation horizon. . 
5. * **evaluation_experiment_time_horizon_4-hours_row_5_table_2** :  
In this evaluation experiment, we evaluate the effect of using time horizon of 4 hours as treatment recommendation horizon.  
6. * **evaluation_experiment_time_horizon_8-hours_row_6_table_2** :  
In this evaluation experiment, we evaluate the effect of using time horizon of 8 hours as treatment recommendation horizon. 
7. * **evaluation_experiment_time_horizon_12-hours_row_7_table_2** :  
In this evaluation experiment, we evaluate the effect of using time horizon of 12 hours as treatment recommendation horizon.  
8. * **evaluation_experiment_time_horizon_24-hours_row_8_table_2** :  
In this evaluation experiment, we evaluate the effect of using time horizon of 24 hours as treatment recommendation horizon.  
9. * **evaluation_experiment_abnormals_accumulated_row_9_table_2** :  
In this evaluation experiment, we evaluate the effect of using accumulated abnormals for finding similar patients.  
10. * **evaluation_experiment_abnormals_instantaneous_row_10_table_2** :  
In this evaluation experiment, we evaluate the effect of using instantaneous abnormals for finding similar patients.  
11. * **evaluation_experiment_similar_patient_set_size_criterion-1_row_11_table_2** :  
In this evaluation experiment, we evaluate the effect of using similarity criterion-1 for selection of similar set.
12. * **evaluation_experiment_similar_patient_set_size_criterion-2_row_12_table_2** :  
In this evaluation experiment, we evaluate the effect of using similarity criterion-2 for selection of similar set.
13. * **evaluation_experiment_similar_patient_set_size_criterion-3_row_13_table_2** :  
In this evaluation experiment, we evaluate the effect of using similarity criterion-3 for selection of similar set.
14. * **evaluation_experiment_similar_patient_set_size_criterion-4_row_14_table_2** :  
In this evaluation experiment, we evaluate the effect of using similarity criterion-2 for selection of similar set.
15. * **testing_experiment_null hypothesis_2-hours_row_1_table_3** :  
In this testing experiment, we test null hyposesis to make treatment prediction for the next 2 hours.
16. * **testing_experiment_datatocare_2-hours_row_2_table_3** :  
In this testing experiment, we test DataToCare to make treatment prediction for the next 2 hours.
17. * **testing_experiment_null hypothesis_4-hours_row_3_table_3** :  
In this testing experiment, we test null hyposesis to make treatment prediction for the next 4 hours.
18. * **testing_experiment_datatocare_4-hours_row_4_table_3** :  
In this testing experiment, we test DataToCare  to make treatment prediction for the next 4 hours.

In each of the experiment subfolders, you will find scripts and text files that will first make features vector and then using that to make training data which will eventually be used to make models for predicting treatments to be given to input patients (500 testing and 300 evaluation patients).

File listed in the experiment sub-folders are scripts: main.py, compute.py, evaluate.py, find_treatments.py, build_state_vectors.py, build_feature_vectors, build_models_predictions.py, helper.py, and csv files: experiment_micu_eval.csv or experiment_micu_testing.csv and valid_admissions_wo_holdout.csv.

Here is the description of each:

* **main.py** :
Main script file that handles the connection to mimiciii-database and calls submodules to execute the variant of the pipeline. 

* **compute.py**:
The script file that using discharge measurement tables computes statistics for numerical using quantiles, all values between the10𝑡ℎ quantile and the90𝑡ℎ quantile as normal; values below the 10𝑡ℎ quantile as low abnormal and values above the 90𝑡ℎ quantile as high abnormal values

* **evaluate.py**:
Script file that using computed statistics first evaluate given patient state and then using patient state and statistics determine all-close patients.

* **find_treatments.py** :
Script file that finds all treatments given to K/all-close patients and returns them as Dataframe. 

* **build_base_vectors.py** :
Script file that first finds all times for which patients have some measurement taken or diagnosis made. Then base vectors are made by incorporating features vectors (demographics, diagnosis, measurements, treatments) with initial values. In last, vectors are partitioned by patients and features type. Partitions are created to enable concurrency through multiprocessing.

* **build_features_vectors.py** :
Script file that has functions that enrich features vectors. Each vectors type (Measurement, Treatment, Demographics, and Diagnosis) is calculated using separate processes for each patient.

* **build_models_predictions.py** :
Script file that processes feature vectors to build a prediction model for each treatment.

* **helper.py** :
Script file that gets and returns measurement type (categorical or numerical) information.

* **cal.py** :
Script file that compiles prediction results and calculates metrics of Precision, Recall, and F1-score of predictions.

* **experiment_micu_eval.csv or experiment_micu_testing.csv** :
A CSV file that contains patients' information for evaluation/testing experiments. subject_id is the unique identifier that specifies an individual patient, hadm_id column is patient admission id and evaltime column is the time we evaluate the patient. admittime is the admission time of the patient and timediff is the time difference in hours between admission and evaluation time. It also contains age since in testing experiments, the patients belong to the hold-out set, therefore the precomputed age information is not found in the valid_admissions_wo_holdout.csv file.

* **valid_admissions_wo_holdout.csv** :
A CSV file that contains the patients who are ever admitted to the MICU. Additionally, it contains the age attribute of patients with respect to specific admission.