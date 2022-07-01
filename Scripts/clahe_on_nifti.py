import cv2 as cv
import numpy as np
from numpy import ndarray
import os

def apply_clahe_on_image(image: ndarray, clahe: cv.CLAHE) -> ndarray:
    return clahe.apply(image.astype(np.uint8),0)   

npy_corrected_and_scaled = os.listdir('../Data/scaled images')
images = []

for path in npy_corrected_and_scaled:
    images.append(np.load('../Data/scaled images/'+path))
   
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

##MODIFY
j = 0
for img in images:
    print(f'Image {j+1}...')
    img_clahe = np.zeros_like(img)
    for i in range(img.shape[0]):
        img_clahe[i,:,:] =  apply_clahe_on_image(img[i,:,:], clahe)
    for i in range(img.shape[1]):
        img_clahe[:,i,:] =  apply_clahe_on_image(img[:,i,:], clahe)
    for i in range(img.shape[2]):
        img_clahe[:,:,i] =  apply_clahe_on_image(img[:,:,i], clahe)
    
    path_to_save = '../Data/preprocessed images/'+npy_corrected_and_scaled[j][:-4]
    j+=1
    np.save(path_to_save, img_clahe) 