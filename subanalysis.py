import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

project_root = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/'
detail = 'apr12_mae_waug_10yr'

test_IDs = pd.read_csv(project_root + "/results/valtest_ID.csv",header=None)
heart_volumes = pd.read_csv(project_root + '/results/heart_sizes.csv',header=None)
expected_age_bins = pd.read_csv(project_root + 'results/age_expected_bin.csv',header=None)
predicted_age_bins = pd.read_csv(project_root + 'results/age_predictions_bin_'+detail+'.csv',header=None)
full_list_gender = pd.read_csv(project_root + 'data/gender.csv',header=None)
full_list_smoke = pd.read_csv(project_root + 'data/smoke.csv',header = None)

test_IDs = np.asarray(test_IDs)
heart_volumes = np.asarray(heart_volumes)
full_list_gender = np.asarray(full_list_gender)
expected_age_bins = np.asarray(expected_age_bins)
predicted_age_bins = np.asarray(predicted_age_bins)
full_list_smoke = np.asarray(full_list_smoke)

test_volumes = []
gender = []
pack_year = []
for id in range(len(test_IDs)):
    for patient in range(len(heart_volumes)):
        if test_IDs[id] == heart_volumes[patient,0]:
            test_volumes.append(heart_volumes[patient,1])
            gender.append(full_list_gender[patient, 1])
            pack_year.append(full_list_smoke[patient,1])
        else:
            continue
        break

test_volumes = np.asarray(test_volumes)
test_data = np.column_stack((test_IDs, test_volumes))
test_data = np.column_stack((test_data, gender))
test_data = np.column_stack((test_data, pack_year))
test_data = np.column_stack((test_data, expected_age_bins))
test_data = np.column_stack((test_data, predicted_age_bins))

bin_diff = []
for id in range(len(test_data)):
    age_diff = test_data[id,4] - test_data[id,5]
    bin_diff.append(age_diff)

test_data = np.column_stack((test_data, bin_diff))

sorted_indices = np.argsort(test_data[:, 1])
sorted_test_data = test_data[sorted_indices]

over_under = []
for i in range(len(sorted_test_data)):
    if sorted_test_data[i,6] < 0:
        over_under.append('over')
    elif sorted_test_data[i,6] == 0:
        over_under.append('accurate')
    else:
        over_under.append('under')

sorted_test_data = np.column_stack((sorted_test_data, over_under))

heartage_dict = {'Patient ID': sorted_test_data[:,0], 'Heart size': sorted_test_data[:,1],'Gender':sorted_test_data[:,2],
                 'Pack/year':sorted_test_data[:,3],'Expected age bin':sorted_test_data[:,4],'Predicted age bin':sorted_test_data[:,5],
                 'Bin difference':sorted_test_data[:,6],'Over-under':sorted_test_data[:,7]}
df = pd.DataFrame(heartage_dict)
df.to_csv(project_root + 'results/sorted_test_data.csv',index=False)