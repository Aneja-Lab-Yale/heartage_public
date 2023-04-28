import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

project_root = '/Users/ckc42/Desktop/thesis/'
detail = 'apr16_mae_waug_10yr'

test_IDs = pd.read_csv(project_root + "results/valtest_ID.csv",header=None)
heart_volumes = pd.read_csv(project_root + 'results/heart_sizes.csv',header=None)
expected_age_bins = pd.read_csv(project_root + 'results/age_expected_bin_'+detail+'.csv',header=None)
predicted_age_bins = pd.read_csv(project_root + 'results/age_predictions_bin_'+detail+'.csv',header=None)
full_list_gender = pd.read_csv(project_root + 'data/gender.csv',header=None)
train_age = pd.read_csv(project_root + 'results/train_age_new.csv',header=None)
test_age = pd.read_csv(project_root + 'results/test_age_new.csv',header=None)

test_IDs = np.asarray(test_IDs)
train_age = np.asarray(train_age)
test_age = np.asarray(test_age)
heart_volumes = np.asarray(heart_volumes)
full_list_gender = np.asarray(full_list_gender)
expected_age_bins = np.asarray(expected_age_bins)
predicted_age_bins = np.asarray(predicted_age_bins)

train_IDs = np.load(project_root+"/results/train_ID_apr4_mae_waug.npy")
train_IDs = np.asarray(train_IDs)
train_data = np.column_stack((train_IDs,train_age))
np.savetxt(project_root + "/results/train_IDs_apr17.csv", train_IDs, delimiter=",",fmt='%s')

test_volumes = []
test_gender = []
train_gender = []


for id in range(len(train_IDs)):
    for patient in range(len(full_list_gender)):
        if train_IDs[id] == full_list_gender[patient,0]:
            train_gender.append(full_list_gender[patient,1])
        else:
            continue
        break

for id in range(len(test_IDs)):
    for patient in range(len(heart_volumes)):
        if test_IDs[id] == heart_volumes[patient,0]:
            test_volumes.append(heart_volumes[patient,1])
            test_gender.append(full_list_gender[patient, 1])
        else:
            continue
        break

train_data = np.column_stack((train_data,train_gender))

test_volumes = np.asarray(test_volumes)
test_data = np.column_stack((test_IDs, test_volumes))
test_data = np.column_stack((test_data, test_gender))
test_data = np.column_stack((test_data, expected_age_bins))
test_data = np.column_stack((test_data, predicted_age_bins))

bin_diff = []
for id in range(len(test_data)):
    age_diff = test_data[id,3] - test_data[id,4]
    bin_diff.append(age_diff)

test_data = np.column_stack((test_data, bin_diff))

sorted_indices = np.argsort(test_data[:, 1])
sorted_test_data = test_data[sorted_indices]

over_under = []
for i in range(len(sorted_test_data)):
    if sorted_test_data[i,5] < 0:
        over_under.append('over')
    elif sorted_test_data[i,5] == 0:
        over_under.append('accurate')
    else:
        over_under.append('under')

sorted_test_data = np.column_stack((sorted_test_data, over_under))
sorted_test_data = np.column_stack((sorted_test_data, test_age))

train_dict = {'PatientID':train_data[:,0],'Expected Age':train_data[:,1],'Gender':train_data[:,2]}
train_df = pd.DataFrame(train_dict)
train_df.to_csv(project_root + 'results/train_data.csv',index=False)

heartage_dict = {'Patient ID': sorted_test_data[:,0], 'Heart size': sorted_test_data[:,1],'Gender':sorted_test_data[:,2],
                 'Expected age bin':sorted_test_data[:,3],'Predicted age bin':sorted_test_data[:,4],
                 'Bin difference':sorted_test_data[:,5],'Over-under':sorted_test_data[:,6],'Age':sorted_test_data[:,7]}
df = pd.DataFrame(heartage_dict)
df.to_csv(project_root + 'results/sorted_test_data.csv',index=False)
