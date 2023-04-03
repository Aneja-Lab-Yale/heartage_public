# Save/Load Images & Metadata to/from Dictionary
# Aneja Lab | Yale School of Medicine
# Enoch Chang
# Created (05/20/20)
# Updated (05/20/20)

"""
Framework for saving image/metadata to dictionary by unique ID
Insert in preprocess pipeline accordingly
"""

# Import
import numpy as np
import pandas as pd


# Folders
DICT_FP = '/Users/enochchang/Desktop/toydict.npy'

#%%
# ---------------------- #
# Save Images +/- Metadata
# ---------------------- #

a = {}  # declare dict for storage
metadata = True  # set True if storing metadata

# example loop over patients/scans
for i in range(3):

	ID = i  # unique identifier (however you extract it)

	# -------------- #
	# preprocess steps resulting in output image
	# -------------- #

	# image = np.zeros((3, 3))  # return of preprocess steps
	image = np.arange(0, 9).reshape(3, 3)  # return of preprocess steps
	metadata1 = 1 * i
	metadata2 = 2 * i

	if metadata:
		# store image and metadata in list
		image_meta = [image, metadata1, metadata2]

		# update dict with image and metadata
		a.update({ID: image_meta})

	else:
		a.update({ID: image})

np.save(DICT_FP, a)


#%%
# # ---------------------- #
# # Load Images +/- Metadata
# # ---------------------- #

b = np.load(DICT_FP, allow_pickle=True).item()


#%%
# initialize df
if metadata:
	columns = ['ID', 'image', 'metadata1', 'metadata2']
else:
	columns = ['ID', 'image']
df = pd.DataFrame(columns=columns)
df.head()


#%%
# fill df
for key in sorted(b):
	b_list = b.get(key)

	if metadata:
		data = pd.DataFrame({'ID': int(key), 'image': [b_list[0]], 'metadata1': int(b_list[1]),
						 'metadata2': int(b_list[2])}, index=[key])
	else:
		data = pd.DataFrame({'ID': int(key), 'image': [b_list]}, index=[key])

	df = df.append(data, ignore_index=True)

#%%
df