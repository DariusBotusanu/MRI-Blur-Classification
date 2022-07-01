from n4biasCorrection import apply_n4bias_correction
import os
import numpy as np
import nibabel as nib

nifti_images_paths = os.listdir('../Data/nifti images')
images = []
for path in nifti_images_paths:
    images.append(nib.load('../Data/nifti images/'+path))
i = 1
for path in nifti_images_paths:
    print(f'Sequence {i}...')
    i+=1
    img = apply_n4bias_correction(input_image = '../Data/nifti images/'+path)
    path_to_save = '../Data/n4 bias corrected/'+path[:-4]
    np.save(path_to_save, img) 
    
    