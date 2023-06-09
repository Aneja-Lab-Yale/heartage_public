# Extracting Patient Data
# ID and Age Data
# Aneja Lab | Yale School of Medicine
# Crystal Cheung
# Created (02/22/23)
# Updated (05/01/23)

# Imports
import pandas as pd
import numpy as np

# local path
project_root = '/Users/ckc42/Desktop/thesis/'

# server path
#project_root = '/home/crystal_cheung/heartage/data/'

# Reads in CT scan excel with patient ages
age_data = pd.read_excel(r"C:\Users\ckc42\Desktop\thesis\CT_scan_data.xlsx")
# Reads in CAC manual segmentation excel with new NLST numbering convention
cac_seg = pd.read_excel(r"C:\Users\ckc42\Desktop\thesis\CAC_ManualSegmentations.xlsx")

# Extracting column data and making into lists
patient_labels = list(age_data["pid"]) #Extracting patient ID column (i.e. 100260)
age_labels = list(age_data["age"]) #Extracting patient age column (i.e. 68)
gender_labels = list(age_data["gender"])
pkyr = list(age_data["pkyr"])
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
# ['NLST_017', 'NLST_022', 'NLST_029', 'NLST_042', 'NLST_069', 'NLST_092', 'NLST_094', 'NLST_105', 'NLST_110',
# 'NLST_117', 'NLST_120', 'NLST_124', 'NLST_138', 'NLST_153', 'NLST_158', 'NLST_161', 'NLST_169', 'NLST_178',
# 'NLST_179', 'NLST_184', 'NLST_188', 'NLST_192', 'NLST_215', 'NLST_226', 'NLST_263', 'NLST_277', 'NLST_280',
# 'NLST_283', 'NLST_284', 'NLST_302', 'NLST_307'] -> scans w/o image-mask dim match
# above indices = [16,21,28,41,68,91,93,104,109,116,119,123,137,152,157,160,168,177,178,183,187,191,214,225,262,276,279,282,283,301,306]
# missing NLST_037 seg [index = 36]
# complied indices = [16,21,28,36,41,68,71,73,90,91,93,95,104,105,109,116,119,123,137,152,157,160,161,168,177,178,183,187,191,195,201,214,218,225,262,276,279,282,283,301,304,306]
bad_scans = [16,21,28,36,41,68,71,73,90,91,93,95,104,105,109,116,119,123,137,152,157,160,161,168,177,178,183,187,191,195,201,214,218,225,262,276,279,282,283,301,304,306]
new_ID_list = [] #new ID list with bad scans removed
corrected_NLST = [] #new list of 3 digit NLST labelling with bad scans removed

# Removes bad scans
for i in range(len(new_number)):
    if i not in bad_scans:
        new_ID_list.append(cac_patientIDs[i])
        corrected_NLST.append(new_number[i])

np.save(project_root + 'data/corrected_NLST.npy', corrected_NLST)

new_age = [] # initialize age list
gender = [] # initialize gender list
smoke = [] # initialize smoking list

# Matches ages from CT data to NLST in CAC segmentation
for j in range(len(new_ID_list)): # for each patient ID in new patient ID list
    for k in range(len(patient_labels)): # for each patient in CT data
        if new_ID_list[j] == patient_labels[k]: # if the NLST_xxxxxx matches between the two

            new_age.append(age_labels[k]) # takes the corresponding CT data patient index to get age and adds to a list of new ages that matches CAC excel
            gender.append(gender_labels[k]) # does the above but with gender
            smoke.append(pkyr[k]) # does the above but with smoking packs per year

            break # skips to outer for loop once found


# creating csv for patient ID and gender
dict = {'Patient ID': corrected_NLST, 'Gender': gender}
df = pd.DataFrame(dict)
df.to_csv(project_root + 'data/gender.csv',index=False)

np.save(project_root + 'data/gender.npy',gender)
np.savetxt(project_root + "ages.csv", new_age, delimiter=",",fmt='%i')
np.save(project_root + 'data/ages.npy',new_age)

age_class = []

# bin legend (5 bins)
# 0 = 50-55
# 1 = 55-60
# 2 = 60-65
# 3 = 65-70

# used for stratifying training and test set in replicate_cnn.py
for i in range(len(new_age)):
    if new_age[i] <= 60:
        age_bin = 0
    elif 60 < new_age[i] <= 65:
        age_bin = 1
    elif 65 < new_age[i] <= 70:
        age_bin = 2
    elif 70 < new_age[i]:
        age_bin = 3

    age_class.append(age_bin)

np.savetxt(project_root + "age_class.csv", age_class, delimiter=",",fmt='%i')
np.save(project_root + 'data/age_class.npy', age_class)