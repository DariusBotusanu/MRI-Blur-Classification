import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

ia.seed(1)

class DataAugmenter():
  def __init__(self):
    self.seq = iaa.Sequential([
                              iaa.Fliplr(0.5), # horizontal flips
                              iaa.Crop(percent=(0, 0.1)), # random crops
                              # Strengthen or weaken the contrast in each image.
                              iaa.LinearContrast((0.75, 1.5)),

                              # Apply affine transformations to each image.
                              # Scale/zoom them, translate/move them, rotate them and shear them.
                              iaa.Affine(
                                  scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                  translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                  rotate=(-25, 25),
                                  shear=(-8, 8)
                                  )
                              ], random_order=True) # apply augmenters in random order
                              
  def augment(self, original_images):
    return self.seq(images=original_images)
  

