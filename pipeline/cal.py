import pandas as pd
import os
import copy
import pickle
import random
import math
import sys
# setting path for importing scripts
sys.path.insert(1, '../common')
import db_handler

conf_interval = 0.9
# maps proportion of values above mean
# to number of standard deviations above mean
# keys will be index / 100 \:[0-9]\.[0-9][0-9]\,
area_to_sd_map = [0.0000, 0.0040, 0.0080, 0.0120, 0.0160, 0.0199, 0.0239, 0.0279, 0.0319, 0.0359, 0.0398, 0.0438, 0.0478, 0.0517, 0.0557, 0.0596, 0.0636, 0.0675, 0.0714, 0.0753, 0.0793, 0.0832, 0.0871, 0.0910, 0.0948, 0.0987, 0.1026, 0.1064, 0.1103, 0.1141, 0.1179, 0.1217, 0.1255, 0.1293, 0.1331, 0.1368, 0.1406, 0.1443, 0.1480, 0.1517, 0.1554, 0.1591, 0.1628, 0.1664, 0.1700, 0.1736, 0.1772, 0.1808, 0.1844, 0.1879, 0.1915, 0.1950, 0.1985, 0.2019, 0.2054, 0.2088, 0.2123, 0.2157, 0.2190, 0.2224, 0.2257, 0.2291, 0.2324, 0.2357, 0.2389, 0.2422, 0.2454, 0.2486, 0.2517, 0.2549, 0.2580, 0.2611, 0.2642, 0.2673, 0.2704, 0.2734, 0.2764, 0.2794, 0.2823, 0.2852, 0.2881, 0.2910, 0.2939, 0.2967, 0.2995, 0.3023, 0.3051, 0.3078, 0.3106, 0.3133, 0.3159, 0.3186, 0.3212, 0.3238, 0.3264, 0.3289, 0.3315, 0.3340, 0.3365, 0.3389, 0.3413, 0.3438, 0.3461, 0.3485, 0.3508, 0.3531, 0.3554, 0.3577, 0.3599, 0.3621, 0.3643, 0.3665, 0.3686, 0.3708, 0.3729, 0.3749, 0.3770, 0.3790, 0.3810, 0.3830, 0.3849, 0.3869, 0.3888, 0.3907, 0.3925, 0.3944, 0.3962, 0.3980, 0.3997, 0.4015, 0.4032, 0.4049, 0.4066, 0.4082, 0.4099, 0.4115, 0.4131, 0.4147, 0.4162, 0.4177, 0.4192, 0.4207, 0.4222, 0.4236, 0.4251, 0.4265, 0.4279, 0.4292, 0.4306, 0.4319, 0.4332, 0.4345, 0.4357, 0.4370, 0.4382, 0.4394, 0.4406, 0.4418, 0.4429, 0.4441, 0.4452, 0.4463, 0.4474, 0.4484, 0.4495, 0.4505, 0.4515, 0.4525, 0.4535, 0.4545, 0.4554, 0.4564, 0.4573, 0.4582, 0.4591, 0.4599, 0.4608, 0.4616, 0.4625, 0.4633, 0.4641, 0.4649, 0.4656, 0.4664, 0.4671, 0.4678, 0.4686, 0.4693, 0.4699, 0.4706, 0.4713, 0.4719, 0.4726, 0.4732, 0.4738, 0.4744, 0.4750, 0.4756, 0.4761, 0.4767, 0.4772, 0.4778, 0.4783, 0.4788, 0.4793, 0.4798, 0.4803, 0.4808, 0.4812, 0.4817, 0.4821, 0.4826, 0.4830, 0.4834, 0.4838, 0.4842, 0.4846, 0.4850, 0.4854, 0.4857, 0.4861, 0.4864, 0.4868, 0.4871, 0.4875, 0.4878, 0.4881, 0.4884, 0.4887, 0.4890, 0.4893, 0.4896, 0.4898, 0.4901, 0.4904, 0.4906, 0.4909, 0.4911, 0.4913, 0.4916, 0.4918, 0.4920, 0.4922, 0.4925, 0.4927, 0.4929, 0.4931, 0.4932, 0.4934, 0.4936, 0.4938, 0.4940, 0.4941, 0.4943, 0.4945, 0.4946, 0.4948, 0.4949, 0.4951, 0.4952, 0.4953, 0.4955, 0.4956, 0.4957, 0.4959, 0.4960, 0.4961, 0.4962, 0.4963, 0.4964, 0.4965, 0.4966, 0.4967, 0.4968, 0.4969, 0.4970, 0.4971, 0.4972, 0.4973, 0.4974, 0.4974, 0.4975, 0.4976, 0.4977, 0.4977, 0.4978, 0.4979, 0.4979, 0.4980, 0.4981, 0.4981, 0.4982, 0.4982, 0.4983, 0.4984, 0.4984, 0.4985, 0.4985, 0.4986, 0.4986, 0.4987, 0.4987, 0.4987, 0.4988, 0.4988, 0.4989, 0.4989, 0.4989, 0.4990, 0.4990]

def sd_to_area(sd):
    sign = 1
    if sd < 0:
        sign = -1
    sd = math.fabs(sd)  # get the absolute value of sd
    index = int(sd * 100)
    if len(area_to_sd_map) <= index:
        return sign * area_to_sd_map[-1] # return last element in array
    if index == (sd * 100):
        return sign * area_to_sd_map[index]
    return sign * (area_to_sd_map[index] + area_to_sd_map[index + 1]) / 2

def area_to_sd(area):
    sign = 1
    if area < 0:
        sign = -1
    area = math.fabs(area)
    for a in range(len(area_to_sd_map)):
        if area == area_to_sd_map[a]:
            return sign * a / 100
        if 0 < a and area_to_sd_map[a - 1] < area and area < area_to_sd_map[a]:
            # our area is between this value and the previous
            # for simplicity, we will just take the sd half way between a - 1 and a
            return sign * (a - .5) / 100
    return sign * (len(area_to_sd_map) - 1) / 100

def mean(grp):
    return sum(grp) / float(len(grp))

def bootstrap(x):
    samp_x = []
    for i in range(len(x)):
        samp_x.append(random.choice(x))
    return samp_x


# compute average and confidence interval for recall, precision and F1-score
def compute_average_ci(sample):
    sample = rc
    observed_mean = mean(sample)

    num_resamples = 10000   # number of times we will resample from our original samples
    num_below_observed = 0   # count the number of bootstrap values below the observed sample statistic
    out = []                # will store results of each time we resample

    for i in range(num_resamples):
        # get bootstrap sample
        # then compute mean
        # append mean to out
        boot_mean = mean(bootstrap(sample))
        if boot_mean < observed_mean:
            num_below_observed += 1
        out.append(boot_mean)

    out.sort()

    # standard confidence interval computations
    tails = (1 - conf_interval) / 2

    # in case our lower and upper bounds are not integers,
    # we decrease the range (the values we include in our interval),
    # so that we can keep the same level of confidence
    lower_bound = int(math.ceil(num_resamples * tails))
    upper_bound = int(math.floor(num_resamples * (1 - tails)))

    # bias-corrected confidence interval computations
    p = num_below_observed / float(num_resamples)   # proportion of bootstrap values below the observed value

    dist_from_center = p - .5   # if this is negative, the original is below the center, if positive, it is above
    z_0 = area_to_sd(dist_from_center)

    # now we want to find the proportion that should be between the mean and one of the tails
    tail_sds = area_to_sd(conf_interval / 2) 
    z_alpha_over_2 = 0 - tail_sds
    z_1_minus_alpha_over_2 = tail_sds

    # in case our lower and upper bounds are not integers,
    # we decrease the range (the values we include in our interval),
    # so that we can keep the same level of confidence
    bias_corr_lower_bound = int(math.ceil(num_resamples * (0.5 + sd_to_area(z_alpha_over_2 + (2 * z_0)))))
    bias_corr_upper_bound =  int(math.floor(num_resamples * (0.5 + sd_to_area(z_1_minus_alpha_over_2 + (2 * z_0)))))


    print(lower_bound)
    print(upper_bound)
    print(bias_corr_lower_bound)
    print(bias_corr_upper_bound)

    print ("Observed mean: %.2f" % observed_mean)
    print ("We have", conf_interval * 100, "% confidence that the true mean", end="")
    print ("is between: %.2f" % out[lower_bound], "and %.2f" % out[upper_bound])

    print ("***** Bias Corrected Confidence Interval *****")
    print ("We have", conf_interval * 100, "% confidence that the true mean", end="")
    print ("is between: %.2f" % out[bias_corr_lower_bound], "and %.2f" % out[bias_corr_upper_bound])


# for each patient, Compile results to output actual and predicted treatment
def compile_results(hadm_id,time,td, tdf_tmp, ver):
    
    print(tdf_tmp.mapped_id.unique().tolist())
    rdf = pd.DataFrame()
    df_pat = pd.DataFrame(columns=['patient','treatment','t','time_diff_from_admission','h-value','actual','predicted'])

    directory = 'results_treat_predict/' + str(hadm_id) + '/'

    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            if 'rf_2' in filename:
                file_path = os.path.join(directory, filename)
                
                tdf = pd.read_pickle(file_path)
            
                rdf = rdf.append(tdf, ignore_index=True)

    if len(rdf) > 0:
        rdf = rdf[rdf.state == 0]
        rdf = rdf[rdf.time == time]

        rdf_tmp_2 = rdf[rdf.predict==1]

    actual_treats = []
    try:
        if not tdf_tmp.empty:
            actual_treats = tdf_tmp.mapped_id.unique().tolist()
            actual_treats = [int(x) for x in actual_treats]
    except:
        pass
    treatments = copy.deepcopy(actual_treats)

    predicted_treats = []
    if len(rdf) > 0:
        predicted_treats = rdf_tmp_2.treatment.unique().tolist()
        predicted_treats = [int(x) for x in predicted_treats]
        treatments.extend(predicted_treats)
        
        print(predicted_treats)
    
    treatments = list(set(treatments))


    if len(treatments) == 0:
        trt_dict = {
            'patient':hadm_id,
            'treatment': 00,
            't':time,
            'time_diff_from_admission':td,
            'h-value':2,
            'actual': 1,
            'predicted' : 1
            }
        df_pat = df_pat.append(trt_dict, ignore_index=True)
    else:
        for treat in treatments:
    
            actual = 0
            if treat in actual_treats:
                actual = 1
                
            predicted = 0
            if treat in predicted_treats:
                predicted = 1
            
            trt_dict = {
            'patient':hadm_id,
            'treatment': treat,
            't':time,
            'time_diff_from_admission':td,
            'h-value':2,
            'actual': actual,
            'predicted' : predicted
            }
            df_pat = df_pat.append(trt_dict, ignore_index=True)

    return df_pat


# get treatment data to determine treatment actually given not from potential treatment set which was given to similar patient 
def get_treatment_data(conn, test_pats):

    treatment_test_pat_data_query = "SELECT hadm_id,starttime,endtime,mapped_id, label "\
                    "FROM mimiciii.inputevents_mv, mimiciii.d3sv1_drugs_mapping "\
                    "WHERE inputevents_mv.itemid = d3sv1_drugs_mapping.itemid AND d3sv1_drugs_mapping.mapping_level = 1 "\
                    "AND hadm_id in ("
    for adm_id in test_pats:
        treatment_test_pat_data_query = treatment_test_pat_data_query + str(adm_id) + ', '
    treatment_test_pat_data_query = ", ".join(treatment_test_pat_data_query.split(", ")[0:-1])
    treatment_test_pat_data_query = treatment_test_pat_data_query + ') '

    treatment_test_pat_df = db_handler.make_selection_query(conn, treatment_test_pat_data_query)

    return treatment_test_pat_df


# calculate metrics for predictions of testing patients
def calculate_results(conn):
    experiment = 'experiment_micu_testing.csv'
    pset = pd.read_csv(experiment)

    tdf = get_treatment_data(conn, pset.hadm_id.tolist())

    tdf['starttime'] = pd.to_datetime(tdf['starttime'])
    tdf['endtime'] = pd.to_datetime(tdf['endtime'])
    df = pd.DataFrame()

    for row in pset.itertuples():
        hadm_id = getattr(row, 'hadm_id')
        time = getattr(row, 'evaltime')
        
        time_horizon = time + pd.Timedelta(2, unit='h')

        td = getattr(row, 'timediff')
        
        tdf_tmp_h = tdf[tdf.hadm_id == hadm_id]
        tdf_tmp_1 = tdf_tmp_h[tdf_tmp_h.starttime >= time]
        tdf_tmp_1 = tdf_tmp_1[tdf_tmp_1.starttime <= time_horizon]
        tdf_tmp_2 = tdf_tmp_h[tdf_tmp_h.starttime <= time]
        tdf_tmp_2 = tdf_tmp_2[tdf_tmp_2.endtime >= time]
        
        tdf_tmp_1 = tdf_tmp_1.append(tdf_tmp_2, ignore_index=True)

        df_pat = compile_results(hadm_id,time,td, tdf_tmp_1,ver)
        if len(df_pat) > 0:
            df = df.append(df_pat, ignore_index=True)


    #process the compile result to calculate precision recall and F1-score of each patient's prediction
    f_scores = []
    percision=[]
    recall=[]

    for hadm_id in df.patient.unique():
        df_pat = df[df.patient==hadm_id]
        pairs = []

        totalactual = len(df_pat[df_pat.actual == 1])
        totalpredicted = len(df_pat[df_pat.predicted == 1])

        for row in df_pat.itertuples():
            actual = getattr(row, 'actual')
            predict = getattr(row, 'predicted')

            pairs.append((actual,predict))

        correctly_predicted = pairs.count((1,1))
        not_predicted = pairs.count((1,0))
        wrong_predicted = pairs.count((0,1)) 

        p=0
        r=0

        if (correctly_predicted+wrong_predicted)>0:
            p = correctly_predicted/(correctly_predicted+wrong_predicted)
        if (correctly_predicted+not_predicted)>0:
            r = correctly_predicted/(correctly_predicted+not_predicted)

        if (totalactual + totalpredicted) >= 1:
            
            percision.append(p)
            recall.append(r)
            if (p + r)>0:
                f = ((2* p* r)/(p + r))
                f_scores.append(f)
            else:
                f_scores.append(0)

    # Precision calculation
    print('Precision Average (Mean) and Confidence Interval')
    compute_average_ci(percision)
    # Recall calculation
    print('Recall Average (Mean) and Confidence Interval')
    compute_average_ci(recall)
    # F1-Score calculation
    print('F1-Score Average (Mean) and Confidence Interval')
    compute_average_ci(f_scores)            


