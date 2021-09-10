# DataToCare
Code Repository of DatatoCare Treatment Prediction Pipeline.  
Please make sure to extract the code in the zip file to the directory/folder "DataToCare", using any other name will break the code as most of the import paths depends on it.

To run DatatoCare Code, see the instructions below.

## Overview
This repository contains the code necessary to run the DataToCare treatment prediction pipeline. 
The patient's current medical state and relevant medical history of former patients are used to identify treatments that should be administered in ICU. 
DataToCare is built upon real-world clinical dataset [MIMIC-III](https://mimic.physionet.org/).

## Requirements
- PostgreSQL 11
- Python 3.8
- UMAP 0.51
- scikit-learn 0.24
- multiprocessing default
- pandas 1.2
- numpy 1.19

## Code Execution

### Setup
Running setup will preprocess the data which involves filtering patients to only those that have only MetaVision data, standardizing the drug data, and determining times for specific diagnosis by examining nurse notes.
To setup:
1) download and build mimiciii with PostgreSQL (https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres)
2) Edit parameters in **./common/databse_connection_parameters.txt** to ensure a connection with the mimiciii-database built in step 1. 
2) run script **./setup/main.py**

### Experiments
Prediction Pipeline
In the experiments folder, you will find different versions of the DataToCare pipeline corresponding to 14 rows of table 2 (evaluation experiments) and 4 rows of table 3 (testing experiments). A brief description of each of the experiments is provided in the readme of the experiments folder along with the information describing each essential subprocess with relevant code scripts used by different pipeline versions.  
Code that will process patient and execute the particular flavor of the pipeline can be found in ./experiments/experiment_name/.   
To execute it on evaluation/testing patients, run script **./experiments/experiment_name/main.py**  
The experiment name also contains the row number and table number with the keywords describing the experiment.