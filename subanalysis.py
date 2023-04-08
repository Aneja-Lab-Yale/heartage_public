import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/'

test_IDs = pd.read_csv(project_root + "/results/valtest_ID.csv")
heart_volumes = pd.read_csv(project_root + '/results/heart_sizes.csv')
test_ages = pd.read_csv(project_root + "/results/valtest_age.csv")
expected_age_bins = pd.read_csv(project_root + 'results/age_expected_bin.csv')

test_IDs = np.asarray(test_IDs)
heart_volumes = np.asarray(heart_volumes)
test_ages = np.asarray(test_ages)

test_volumes = []
for id in range(len(test_IDs)):
    for patient in range(len(heart_volumes)):
        if test_IDs[id] == heart_volumes[patient,0]:
            test_volumes.append(heart_volumes[patient,1])
        else:
            continue
        break

