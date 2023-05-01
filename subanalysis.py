# Heart Age Analysis
# Aneja Lab | Yale School of Medicine
# Crystal Cheung
# Created (03/01/23)
# Updated (05/01/23)

# Imports
import pandas as pd
import numpy as np

# local path
project_root = '/Users/ckc42/Desktop/thesis/'
detail = 'apr16_mae_waug_10yr'

# loading files for analysis
test_IDs = pd.read_csv(project_root + "results/valtest_ID.csv",header=None) # test patient IDs
heart_volumes = pd.read_csv(project_root + 'results/heart_sizes.csv',header=None) # patient heart sizes
expected_age_bins = pd.read_csv(project_root + 'results/age_expected_bin_'+detail+'.csv',header=None) # expected age bins
predicted_age_bins = pd.read_csv(project_root + 'results/age_predictions_bin_'+detail+'.csv',header=None) # predicted age bins
full_list_gender = pd.read_csv(project_root + 'data/gender.csv',header=None) # patient genders
full_list_smoke = pd.read_csv(project_root + 'data/smoke.csv',header=None) # patient smoking packs per year
train_age = pd.read_csv(project_root + 'results/train_age_new.csv',header=None) # training patient ages
test_age = pd.read_csv(project_root + 'results/test_age_new.csv',header=None) # test patient ages
predicted_age = pd.read_csv(project_root + 'results/age_predictions_reg_'+detail+'.csv',header=None) # predicted patient ages

# setting above lists as arrays
test_IDs = np.asarray(test_IDs)
train_age = np.asarray(train_age)
test_age = np.asarray(test_age)
heart_volumes = np.asarray(heart_volumes)
full_list_gender = np.asarray(full_list_gender)
full_list_smoke = np.asarray(full_list_smoke)
expected_age_bins = np.asarray(expected_age_bins)
predicted_age_bins = np.asarray(predicted_age_bins)
predicted_age = np.asarray(predicted_age)

# loading training patient ID
train_IDs = np.load(project_root+"/results/train_ID_apr4_mae_waug.npy")
train_IDs = np.asarray(train_IDs)

# initializing demographics lists
test_volumes = []
test_gender = []
test_smoke = []
train_volumes = []
train_gender = []
train_smoke = []

# get demographics for training patients
for id in range(len(train_IDs)):
    for patient in range(len(heart_volumes)):
        if train_IDs[id] == heart_volumes[patient,0]:
            train_volumes.append(heart_volumes[patient,1])
            train_gender.append(full_list_gender[patient,1])
            train_smoke.append(full_list_smoke[patient,1])
        else:
            continue
        break

# get demographics for test patients
for id in range(len(test_IDs)):
    for patient in range(len(heart_volumes)):
        if test_IDs[id] == heart_volumes[patient,0]:
            test_volumes.append(heart_volumes[patient,1])
            test_gender.append(full_list_gender[patient, 1])
            test_smoke.append(full_list_smoke[patient,1])
        else:
            continue
        break

# making array of training demographics data
train_volumes = np.asarray(train_volumes)
train_data = np.column_stack((train_IDs, train_age))
train_data = np.column_stack((train_data, train_volumes))
train_data = np.column_stack((train_data, train_gender))
train_data = np.column_stack((train_data, train_smoke))

# making array of test demographics data
test_volumes = np.asarray(test_volumes)
test_data = np.column_stack((test_IDs, test_volumes))
test_data = np.column_stack((test_data, test_gender))
test_data = np.column_stack((test_data, test_smoke))
test_data = np.column_stack((test_data, expected_age_bins))
test_data = np.column_stack((test_data, predicted_age_bins))

# calculating the difference between expected and predicted bins
bin_diff = []
for id in range(len(test_data)):
    age_diff = test_data[id,4] - test_data[id,5]
    bin_diff.append(age_diff)

test_data = np.column_stack((test_data, bin_diff)) # adding bin differences to test data
test_data = np.column_stack((test_data, test_age)) # adding patient true age to test data
test_data = np.column_stack((test_data, predicted_age)) # adding patient predicted age to test data

# sorts data by heart size, smallest to largest
sorted_indices = np.argsort(test_data[:, 1])
sorted_test_data = test_data[sorted_indices]

# takes bin differences and assigns it as over (pred > exp), accurate, or under (pred < exp)
over_under = []
for i in range(len(sorted_test_data)):
    if sorted_test_data[i,6] < 0:
        over_under.append('over')
    elif sorted_test_data[i,6] == 0:
        over_under.append('accurate')
    else:
        over_under.append('under')

# adding over_under assignments to test data array
sorted_test_data = np.column_stack((sorted_test_data, over_under))

# making a dictionary csv of training demographics
train_dict = {'PatientID':train_data[:,0],'Expected Age':train_data[:,1],'Heart size':train_data[:,2],
              'Gender':train_data[:,3],'Smoke':train_data[:,4]}
train_df = pd.DataFrame(train_dict)
train_df.to_csv(project_root + 'results/train_data.csv',index=False)

# making a dictionary csv of testing demographics (+ age bins and ages)
heartage_dict = {'Patient ID': sorted_test_data[:,0], 'Heart size': sorted_test_data[:,1],'Gender':sorted_test_data[:,2],
                 'Smoke':sorted_test_data[:,3],'Expected age bin':sorted_test_data[:,4],'Predicted age bin':sorted_test_data[:,5],
                 'Bin difference':sorted_test_data[:,6],'Expected Age':sorted_test_data[:,7],
                 'Predicted Age':sorted_test_data[:,8],'Over-under':sorted_test_data[:,9]}
df = pd.DataFrame(heartage_dict)
df.to_csv(project_root + 'results/sorted_test_data.csv',index=False)
