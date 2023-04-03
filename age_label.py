# Extracting Patient Data
# ID and Age Data
# Aneja Lab | Yale School of Medicine
# Crystal Cheung
# Created (02/22/23)
# Updated (02/23/23)

# Imports
import pandas as pd
import numpy as np

project_root = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/'
#project_root = '/home/crystal_cheung/heartage/data/'

# Reads in CT scan excel with patient ages
age_data = pd.read_excel(r"C:\Users\Crystal\Desktop\College\PMAE\Thesis\CT_scan_data.xlsx")
# Reads in CAC manual segmentation excel with new NLST numbering convention
cac_seg = pd.read_excel(r"C:\Users\Crystal\Desktop\College\PMAE\Thesis\CAC_ManualSegmentations.xlsx")

# Extracting column data and making into lists
patient_labels =  list(age_data["pid"]) #Extracting patient ID column (i.e. 100260)
age_labels = list(age_data["age"]) #Extracting patient age column (i.e. 68)
cac_patientIDs = list(cac_seg["patientID"]) #Extracting patient ID column (i.e. NLST_100260)
new_number = list(cac_seg["Number Key"]) #Extracting new NLST numbering convention (i.e. 3)

# Renaming patient IDs as NLST_xxxxxx
for i in range(len(patient_labels)):
    patient_labels[i] = "NLST_" + str(patient_labels[i])

# Renaming new NLST numbering convention
for i in range(len(new_number)):
    new_number[i] = "%03d" % new_number[i] #sets numbers to have 3 digits to match s3 naming convention (i.e. 1 = 001)
    new_number[i] = "NLST_" + str(new_number[i]) #renames NLST_xxxxxx as NLST_xxx as notated in CAC segmentation excel

#bad_scans = [71,73,90,95,105,161,195,201,218,304] #indices of bad scans in cac segmentation list
# NLST_017 NLST_022 NLST_029 NLST_042 NLST_094 NLST_110 NLST_158 NLST_169 NLST_192 -> scans w/o image-mask dim match
# above indices = [16,21,28,41,93,109,157,168,191]
bad_scans = [16,21,28,36,41,51,63,65,67,68,69,70,71,73,74,75,76,77,79,81,82,83,85,86,87,90,91,93,94,95,96,97,98,99,100,103,104,105,106,107,109,113,115,116,117,119,120,122,123,124,125,126,127,131,132,133,134,135,136,137,138,139,142,144,145,147,148,149,151,152,153,154,155,156,157,158,159,160,161,163,164,166,168,169,170,176,177,178,179,180,183,184,185,186,187,190,191,192,193,195,197,198,199,200,201,203,205,207,208,209,210,211,213,214,215,216,218,219,221,223,224,225,226,227,228,229,230,231,232,233,235,236,237,241,243,244,245,246,248,249,250,251,252,253,255,256,257,259,261,262,263,264,265,266,267,268,269,270,271,272,273,274,276,277,279,280,282,283,284,285,286,289,290,291,292,294,298,299,300,301,302,304,305,306]
new_ID_list = [] #new ID list with bad scans removed
corrected_NLST = [] #new list of 3 digit NLST labelling with bad scans removed

print(len(bad_scans))
print(len(new_number))
# Removes bad scans
for i in range(len(new_number)):
    if i not in bad_scans:
        new_ID_list.append(cac_patientIDs[i])
        corrected_NLST.append(new_number[i])

print(len(corrected_NLST))
np.save(project_root + 'corrected_NLST.npy', corrected_NLST)
#print(new_ID_list[71])
#print(corrected_NLST[71])
#print(len(new_ID_list))
#print(len(corrected_NLST))

new_age = [] #initialize age list

# Matches ages from CT data to NLST in CAC segmentation
for j in range(len(new_ID_list)): # for each patient ID in new patient ID list
    for k in range(len(patient_labels)): # for each patient in CT data
        if new_ID_list[j] == patient_labels[k]: # if the NLST_xxxxxx matches between the two

            new_age.append(age_labels[k]) #takes the corresponding CT data patient index to get age and adds to a list of new ages that matches CAC excel
            break # skips to outer for loop once found

np.savetxt(project_root + "ages.csv", new_age, delimiter=",",fmt='%i')
np.save(project_root + 'data/ages.npy',new_age)


age_class = []

# bin legend (5 bins)
# 0 = 50-55
# 1 = 55-60
# 2 = 60-65
# 3 = 65-70
# 4 = 70-75
# 5 = 75-80

for i in range(len(new_age)):
    if new_age[i] < 60:
        age_bin = 0
    elif 60 <= new_age[i] < 65:
        age_bin = 1
    elif 65 <= new_age[i] < 70:
        age_bin = 2
    elif 70 <= new_age[i]:
        age_bin = 3

    age_class.append(age_bin)

np.savetxt(project_root + "data/age_class.csv", age_class, delimiter=",",fmt='%i')
np.save(project_root + 'data/age_class.npy', age_class)
#print(len(new_age))
#print(corrected_NLST[71])
#print(new_ID_list[71])
#print(new_age[71])

apr5_test = np.load("C:/Users/Crystal/Downloads/test_ID_apr3_5.npy")
apr5_val = np.load("C:/Users/Crystal/Downloads/val_ID_apr3_5.npy")
apr5_train = np.load("C:/Users/Crystal/Downloads/train_ID_apr3_5.npy")
test_age = []
train_age = []
val_age = []
for j in range(len(apr5_test)): # for each patient ID in new patient ID list
    for k in range(len(corrected_NLST)): # for each patient in CT data
        if apr5_test[j] == corrected_NLST[k]: # if the NLST_xxxxxx matches between the two

            test_age.append(new_age[k]) #takes the corresponding CT data patient index to get age and adds to a list of new ages that matches CAC excel
            break # skips to outer for loop once found

for j in range(len(apr5_val)): # for each patient ID in new patient ID list
    for k in range(len(corrected_NLST)): # for each patient in CT data
        if apr5_val[j] == corrected_NLST[k]: # if the NLST_xxxxxx matches between the two

            val_age.append(new_age[k]) #takes the corresponding CT data patient index to get age and adds to a list of new ages that matches CAC excel
            break # skips to outer for loop once found

for j in range(len(apr5_train)): # for each patient ID in new patient ID list
    for k in range(len(corrected_NLST)): # for each patient in CT data
        if apr5_train[j] == corrected_NLST[k]: # if the NLST_xxxxxx matches between the two

            train_age.append(new_age[k]) #takes the corresponding CT data patient index to get age and adds to a list of new ages that matches CAC excel
            break # skips to outer for loop once found

np.savetxt(project_root + "data/test_age.csv", test_age, delimiter=",",fmt='%i')
np.savetxt(project_root + "data/train_age.csv", train_age, delimiter=",",fmt='%i')
np.savetxt(project_root + "data/val_age.csv", val_age, delimiter=",",fmt='%i')