import SimpleITK as sitk
import numpy as np
import nibabel as nib

def apply_n4bias_correction(
    input_image : nib.nifti1.Nifti1Image
) -> np.array:
    
    
    inputImage = sitk.ReadImage(input_image, sitk.sitkFloat32)
    image = inputImage
    
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
   
    corrector.Execute(image, maskImage)
    
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

    corrected_image_full_resolution = inputImage / sitk.Exp( log_bias_field )
    
    return  np.transpose(sitk.GetArrayFromImage(corrected_image_full_resolution))