import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.signal import convolve2d
  
def model_confusion_matrix(model, test_path, transform=False, kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])):
  df = pd.read_csv(test_path)
  if 'partition' in df.columns:
    df = df[df['partition'] == 'test']
  true_labels = df['label']

  X = np.empty((len(df['path']),256, 256))
  # Generate data
  for i, ID in enumerate(list(df['path'])):
    if transform:
      X[i,] = convolve2d(np.load(ID), kernel, mode='same')
    else:
      X[i,] = np.load(ID)  
  
  predictions = model.predict(X)
  predictions = np.array(predictions, dtype=int)
  return confusion_matrix(true_labels, predictions)

def get_mislabeled_images(model, test_path):
  df = pd.read_csv(test_path)
  if 'partition' in df.columns:
    df = df[df['partition'] == 'test']
  true_labels = df['label']

  X = np.empty((len(df['path']),256, 256))
  # Generate data
  for i, ID in enumerate(list(df['path'])):
    X[i,] = np.load(ID)  

  predictions = model.predict(X)
  predictions = np.array(predictions, dtype=int)

  difference = np.array(true_labels).reshape(predictions.shape)-predictions
  difference = difference.reshape(difference.shape[0])
  mislabeled_images = []
  mislabeled_paths = []
  for i in range(len(difference)):
    if difference[i] !=0:
      mislabeled_images.append(X[i])
      mislabeled_paths.append(df['path'].iloc[i])
      
  return mislabeled_images, mislabeled_paths


def confusion_mat_to_df(confusion_mat):
  df = pd.DataFrame()
  df['nomotion'] = [confusion_mat[0][0], confusion_mat[1][0]]
  df['motion'] = [confusion_mat[0][1], confusion_mat[1][1]]
  df.index = ['nomotion','motion']
  return df

def specificity(confusion_mat):
  '''
  The (# true "motion")/[(# true "motion")+(# false "motion")]
  '''
  return confusion_mat[1][1]/(confusion_mat[1][1]+confusion_mat[0][1])

def recall(confusion_mat):
  '''
  The (# true "nomotion")/[(# true "nomotion")+(# false "motion")]
  '''
  return confusion_mat[0][0]/(confusion_mat[0][0]+confusion_mat[0][1])

def print_model_metrics(model, test_path, transform=False, kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])):
  confusion_mat = model_confusion_matrix(model, test_path, transform=False, kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]]))
  confusion_frame = confusion_mat_to_df(confusion_mat)
  model_specificity = specificity(confusion_mat)
  model_recall = recall(confusion_mat)
  
  print(f'Specificity = {model_specificity}')
  print(f'Recall = {model_recall}')
  print("Confusion matrix:")
  print(confusion_frame)
  return model_specificity, model_recall, confusion_mat