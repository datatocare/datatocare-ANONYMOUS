# DataToCare-ANONYMOUS
Anonymous version of DatatoCare Treatment Prediction Pipeline.

To run DatatoCare, see instructions below.

## Overview
This repository contains code necessary to run DataToCare treatment prediction pipeline. 
Patients current medical state and relevant medical history of former patients are used to identify treatment that should be administered in ICU.
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
Running setup will preprocessed the data which involves filtering patients to only those that have only MetaVision data, standardizing the drug data and determining time for specific diagnosis by examining nurse notes.
To setup:
1) download and build mimiciii with postgresql (https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres)
2) run script **./setup/main.py**

### Prediction Pipeline
Code that will process patient and execute pipeline can be found in ./pipeline/. The pipeline directory also contains the information describing each subprocesses accompained by relveant code scripts. 
To execute pipeline on testing pateints, run script **./pipeline/main.py**








