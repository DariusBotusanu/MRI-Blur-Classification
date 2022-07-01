from numpy import ndarray
import numpy as np
import os


#MIGHT NEED INTEGER CAST
def scale_to_0_255(image: ndarray)->ndarray:
    '''
    min -> the lowest value of a pixel
    max -> the highest value of a pixel
    maps [min,max] -> [0,255]
    the function is output = m*image+b
    it is actually a line joining (min,0) and (max,255)
    '''
    min_pixel = image.min()
    max_pixel = image.max()
    m = 255/(max_pixel-min_pixel)
    b = -m*min_pixel
    return m*image+b

if __name__ == "__main__":
    npy_bias_corrected = os.listdir('../Data/n4 bias corrected')
    images = []
    for path in npy_bias_corrected:
        images.append(np.load('../Data/n4 bias corrected/'+path))
    
    for i in range(len(images)):
        print(f'Scaling image {i+1}...')
        scaled_image = scale_to_0_255(images[i])
        path_to_save = '../Data/scaled images/'+npy_bias_corrected[i][:-4]
        np.save(path_to_save, scaled_image) 
    