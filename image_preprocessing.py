# Heart Age Image Preprocessing
# Aneja Lab | Yale School of Medicine
# Crystal Cheung
# Created (01/20/23)
# Updated (02/23/23)

# Import
# importing tensorflow software lib as variable tf (for machine learning)
import numpy as np
# importing numpy software lib as variable np (for math fx)
# adds system path to this file location (ex: /Users/sanjayaneja/Desktop/Dropbox/Sanjay/Code/aneja_lab_common)
# from Useful_Functions.Misc_Functions import listdir_nohidden
# from Useful_Functions.Misc_Functions import listdir_dicom
# from Keras_Miscellaneous.Keras_Callbacks import callbacks_model as cb
from dipy.io.image import load_nifti
import os
import math
# import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import statistics
import pandas as pd

project_root = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/'
image_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Whole_CT/'
mask_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Heart_segmentations/'
# image_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Image/'
# mask_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Mask/'

# project_root = '/home/crystal_cheung/'
# image_path = '/home/crystal_cheung/Whole_CT/'
# mask_path = '/home/crystal_cheung/Heart_segmentations/'

# -------------------------------------------------- MODEL SETUP -------------------------------------------------------


# patient_IDs = ['NLST_100570','NLST_120701','NLST_200129', 'NLST_202822'] #patient id lists
# patient_IDs = corrected_NLST # corrected_NLST from age_label.py
patient_IDs = list(np.load(project_root + 'data/corrected_NLST.npy'))
# age_labels = [63,67,58,57] #age list
# age_labels = new_age #new age list from age_label.py

#patient_list = [] # sets up list of overlaid image and mask array
#patient_shape = [] # sets up list of shapes of overlay arrays

#i_dim = []
#m_dim = []
#bad_IDs = []

patient_list = []
patient_shape = []
heart_size = []

for i,patient in enumerate(patient_IDs):
# for i in range(20):
    # getting image and masks of each patients from files
    image, affine, voxsize, coords = load_nifti(os.path.join(image_path, patient_IDs[i] + '.nii.gz'), return_voxsize = True, return_coords=True)

    mask, affine2, voxsize2, coords2 = load_nifti(os.path.join(mask_path,'Heart_'+ patient_IDs[i] + '.nii.gz'), return_voxsize = True, return_coords=True)

    #i_shape = image.shape
    #m_shape = mask.shape

    #i_dim.append(i_shape)
    #m_dim.append(m_shape)

    #if i_shape != m_shape:
        #bad_IDs.append(patient_IDs[i])

    # sets voxel size and other image features to be the same in the mask and image
    assert image.shape == mask.shape
    # assert (affine == affine2).all()
    shape = np.array(image.shape)

    #make image and mask into arrays and overlay them to just get the heart
    image_array = np.array(image) #make array from whole CT
    mask_array = np.array(mask) #make array from total heart segment mask
    overlay = np.multiply(image_array,mask_array) #overlays the two to get just the total heart area of image
    nonzero_voxels = np.count_nonzero(overlay)
    volume = voxsize[0] * voxsize[1] * voxsize[2] * nonzero_voxels
    heart_size.append(volume)

    #put patient arrays into a list
    patient_list.append(overlay.astype('float32')) #puts patient overlays into list [array1 array2 ...]
    patient_shape.append(overlay.shape) #puts patient overlay shapes into list [array1_shape array2_shape ...]

dict = {'Patient ID': patient_IDs, 'Heart size': heart_size}
df = pd.DataFrame(dict)
df.to_csv(project_root + '/results/heart_sizes.csv',index=False)

#print(patient_shape)

#ax = fig.add_subplot(1, 4, 1)
#imgplot = plt.imshow(patient_list[0][:, :, 28], cmap="gray")
#ax.set_title('28')
#ax = fig.add_subplot(1, 4, 2)
#imgplot = plt.imshow(patient_list[0][:, :, 29], cmap="gray")
#ax.set_title('29')
#ax = fig.add_subplot(1, 4, 3)
#imgplot = plt.imshow(patient_list[0][:, :, 81], cmap="gray")
#ax.set_title('81')
#ax = fig.add_subplot(1, 4, 4)
#imgplot = plt.imshow(patient_list[0][:, :, 82], cmap="gray")
#ax.set_title('82')

cropped = []
cropped_shape = []

for i in range(len(patient_list)): #loop through each patient overlay array
#for i in range(20):
    crop_indices = np.nonzero(patient_list[i])
    #heart_section = [*set(crop_indices[2])]
    start_depth = min(crop_indices[2])
    end_depth = max(crop_indices[2])

    crop_img = patient_list[i][:,:, start_depth:end_depth+1]
    cropped.append(crop_img)
    cropped_shape.append(crop_img.shape)


#print(cropped_shape)

#fig2 = plt.figure()
#plt.imshow(patient_list[54][100:500,100:500,39],cmap='gray')

#fig = plt.figure()
#ax = fig.add_subplot(1, 4, 1)
#imgplot = plt.imshow(cropped[0][:, :, 25], cmap="gray")
#ax.set_title('0')
#ax = fig.add_subplot(1, 4, 2)
#imgplot = plt.imshow(cropped[1][:, :, 25], cmap="gray")
#ax.set_title('1')
#ax = fig.add_subplot(1, 4, 3)
#imgplot = plt.imshow(cropped[2][:, :, 40], cmap="gray")
#ax.set_title('2')
#ax = fig.add_subplot(1, 4, 4)
#imgplot = plt.imshow(cropped[3][:, :, 25], cmap="gray")
#ax.set_title('3')

#getting the biggest patient array dimensions
#z_max = max(cropped_shape) #gets largest dimensions out of list of overlays [in this case: (512x512x264)]
z_median = statistics.median(cropped_shape)
patient_pad = [] #makes new list for padded patient arrays
patient_pad_shape = [] #makes new list for shape of padded patient arrays
#print(z_max) #see z_max

for i,patient in enumerate(patient_IDs):
#for i in range(20):
    #keeps the patient array that has the max dimensions
    if cropped_shape[i] == z_median: #if patient has largest dimensions, keep that array and shape
        patient_pad.append(cropped[i]) #adds that patient array to pad list
        patient_pad_shape.append(z_median) #adds that patient shape to pad shape list

    elif cropped_shape[i] > z_median:

    # finding the center index for x,y,z axes of the starting and final overlay
        center_slice = math.floor(cropped_shape[i][2]/2)
        start_slice = center_slice - math.floor(z_median[2]/2)
        final_slice = center_slice + math.ceil(z_median[2]/2)

        second_crop = cropped[i][:,:,start_slice:final_slice]
        patient_pad.append(second_crop)
        patient_pad_shape.append(second_crop.shape)

    elif cropped_shape[i] < z_median:
        padding_value = (z_median[2] - cropped_shape[i][2]) / 2 #gets number of pad slices needed on the top and bottom of patient array

        #pads top and bottom separately
        pad_top = np.zeros((z_median[0], z_median[0], math.ceil(padding_value)),dtype=np.float32) #sets top padding values (x,y,z) [accounts for odd padding values]
        pad_below = np.zeros((z_median[0], z_median[0], math.floor(padding_value)),dtype=np.float32) #sets bottom padding values (x,y,z) [accounts for odd padding values]

        #makes new list of padded patient arrays
        pad_overlay = np.concatenate((pad_top, cropped[i], pad_below), axis=2) #puts top pad array, patient array, bottom pad array together into one pad array
        patient_pad.append(pad_overlay) #adds padded patient arrays to list
        patient_pad_shape.append(pad_overlay.shape) #adds padded patient array shapes to list

#print(patient_pad_shape)

final = [] #list of downsampled arrays
final_shape = [] #list of downsampled array shapes
final_img_length = 60
final_img_slice = 47

# resampling images
for i,patient in enumerate(patient_IDs):
    resample = zoom(patient_pad[i], (final_img_length/z_median[0], final_img_length/z_median[0],1)) #gives array (128,128,66), downsamples pixels
    #resample = zoom(patient_pad[i], (121/512,121/512,145/254)) #gives array (121, 121, 145)
    final.append(resample) #adds resampled arrays to "final" list
    final_shape.append(resample.shape) #adds resampled array shapes to "final_shape" list

# (optional) if cropping of actual images needed, gets list of cropped patient arrays and the wanted shape
#cropped = []
#cropped_shape = []

#for i, patient in enumerate(patient_pad):

    #the starting image dimensions
    #start_img_length = z_max[0]
    #start_img_slice = z_max[2]

    #finding the center index for x,y,z axes of the starting and final overlay
    #center = int(start_img_length / 2)
    #final_half = int(final_img_length / 2)
    #center_slice = math.floor(start_img_slice / 2)
    #final_slice = math.floor(final_img_slice / 2)

    #find the start and end indices of arrays for cropping
    #start_row = center - final_half
    #start_col = start_row
    #end_row = start_row + final_img_length
    #end_col = end_row
    #start_depth = center_slice - final_slice
    #end_depth = start_depth + final_img_slice

    #sets the indices to make new cropped arrays
    #crop_img = final[i][start_row:end_row, start_col:end_col, start_depth:end_depth]
    #cropped.insert(i, crop_img)
    #cropped_shape.insert(i, crop_img.shape)

#print(cropped_shape)

#plots original image, original mask, overlay, padded overlay, downsampled overlay
#fig = plt.figure('test.jpg')
#ax = fig.add_subplot(1, 5, 1)
#imgplot = plt.imshow(image[:, :, 64], cmap="gray")
#ax.set_title('Original')
#ax = fig.add_subplot(1, 5, 2)
#imgplot = plt.imshow(mask[:, :, 64], cmap="gray")
#ax.set_title('Mask')
#ax = fig.add_subplot(1, 5, 3)
#imgplot = plt.imshow(overlay[:, :, 64], cmap="gray")
#ax.set_title('Overlay')
#ax = fig.add_subplot(1, 5, 4)
#imgplot = plt.imshow(pad_overlay[:, :, 112], cmap="gray")
#ax.set_title('Padded Overlay')
#ax = fig.add_subplot(1, 5, 5)
#imgplot = plt.imshow(downsample[:, :, 14], cmap="gray")
#ax.set_title('Cropped Overlay')
#fig.savefig('test.jpg')
#print(patient_shape)
#print(cropped_shape)
#print(z_max)
#print(patient_pad_shape)
#print(final_shape)

np.save(project_root + 'data/final_images.npy', final)
print('done')


