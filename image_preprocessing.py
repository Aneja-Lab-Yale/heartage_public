# Heart Age Image Preprocessing
# Aneja Lab | Yale School of Medicine
# Crystal Cheung
# Created (01/20/23)
# Updated (05/01/23)

# Import
import numpy as np
from dipy.io.image import load_nifti
import os
import math
# import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import statistics
import pandas as pd

# local paths
# project_root = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/'
# image_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Whole_CT/'
# mask_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Heart_segmentations/'
# image_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Image/'
# mask_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Mask/'

# server paths
# project_root = '/home/crystal_cheung/'
# image_path = '/home/crystal_cheung/Whole_CT/'
# mask_path = '/home/crystal_cheung/Heart_segmentations/'

# -------------------------------------------------- MODEL SETUP -------------------------------------------------------

patient_IDs = list(np.load(project_root + 'data/corrected_NLST.npy'))

patient_list = []
patient_shape = []
heart_size = []

for i,patient in enumerate(patient_IDs):
    # getting image and masks of each patient's from files
    image, affine, voxsize, coords = load_nifti(os.path.join(image_path, patient_IDs[i] + '.nii.gz'), return_voxsize = True, return_coords=True)

    mask, affine2, voxsize2, coords2 = load_nifti(os.path.join(mask_path,'Heart_'+ patient_IDs[i] + '.nii.gz'), return_voxsize = True, return_coords=True)

    # sets voxel size and other image features to be the same in the mask and image
    assert image.shape == mask.shape
    shape = np.array(image.shape)

    # make image and mask into arrays and overlay them to just get the heart
    image_array = np.array(image) # make array from whole CT
    mask_array = np.array(mask) # make array from total heart segment mask
    overlay = np.multiply(image_array,mask_array) # overlays the two to get just the total heart area of image
    nonzero_voxels = np.count_nonzero(overlay) # count number of non-zero voxels
    volume = voxsize[0] * voxsize[1] * voxsize[2] * nonzero_voxels # calculate volume from nonzero voxels and voxel size
    heart_size.append(volume) # add volume to a list of heart sizes

    # put patient arrays into a list
    patient_list.append(overlay.astype('float32')) # puts patient overlays into list [array1 array2 ...]
    patient_shape.append(overlay.shape) # puts patient overlay shapes into list [array1_shape array2_shape ...]

# creating and saving csv of heart sizes for all patients
dict = {'Patient ID': patient_IDs, 'Heart size': heart_size}
df = pd.DataFrame(dict)
df.to_csv(project_root + '/results/heart_sizes.csv',index=False)

# set list for images after cropping
cropped = []
cropped_shape = []

for i in range(len(patient_list)): # loop through each patient overlay array
    crop_indices = np.nonzero(patient_list[i]) # get indices for heart section
    start_depth = min(crop_indices[2]) # starting index of heart slices
    end_depth = max(crop_indices[2]) # ending index of heart slices

    crop_img = patient_list[i][:,:, start_depth:end_depth+1] # crops image to just heart
    cropped.append(crop_img) # adds image to cropped list
    cropped_shape.append(crop_img.shape) # adds image shape to cropped shape list

# getting the median patient array dimensions
z_median = statistics.median(cropped_shape)
patient_pad = [] # makes new list for padded patient arrays
patient_pad_shape = [] # makes new list for shape of padded patient arrays

for i,patient in enumerate(patient_IDs):
    # keeps the patient array that has the median dimensions
    if cropped_shape[i] == z_median: # if patient has median dimensions, keep that array and shape
        patient_pad.append(cropped[i]) # adds that patient array to pad list
        patient_pad_shape.append(z_median) # adds that patient shape to pad shape list

    # if patient array is larger than median shape, crop
    elif cropped_shape[i] > z_median:

        # finding the center index for x,y,z axes of the starting and final overlay
        center_slice = math.floor(cropped_shape[i][2]/2)
        start_slice = center_slice - math.floor(z_median[2]/2)
        final_slice = center_slice + math.ceil(z_median[2]/2)

        # cropping array to median shape and adding it and shape to lists
        second_crop = cropped[i][:,:,start_slice:final_slice]
        patient_pad.append(second_crop)
        patient_pad_shape.append(second_crop.shape)

    # if patient array is less than median shape, pad
    elif cropped_shape[i] < z_median:
        padding_value = (z_median[2] - cropped_shape[i][2]) / 2 # gets number of pad slices needed on the top and bottom of patient array

        # pads top and bottom separately
        pad_top = np.zeros((z_median[0], z_median[0], math.ceil(padding_value)),dtype=np.float32) # sets top padding values (x,y,z) [accounts for odd padding values]
        pad_below = np.zeros((z_median[0], z_median[0], math.floor(padding_value)),dtype=np.float32) # sets bottom padding values (x,y,z) [accounts for odd padding values]

        # makes new list of padded patient arrays
        pad_overlay = np.concatenate((pad_top, cropped[i], pad_below), axis=2) # puts top pad array, patient array, bottom pad array together into one pad array
        patient_pad.append(pad_overlay) # adds padded patient arrays to list
        patient_pad_shape.append(pad_overlay.shape) # adds padded patient array shapes to list

final = [] # list of downsampled arrays
final_shape = [] # list of downsampled array shapes
# final image shape and slice (slice derived from median shape above)
final_img_length = 60
final_img_slice = 47

# resampling images
for i,patient in enumerate(patient_IDs):
    # gives array (60,60,47), downsamples pixels
    resample = zoom(patient_pad[i], (final_img_length/z_median[0], final_img_length/z_median[0],1))
    final.append(resample) # adds resampled arrays to "final" list
    final_shape.append(resample.shape) # adds resampled array shapes to "final_shape" list

np.save(project_root + 'data/final_images.npy', final)
print('done')


