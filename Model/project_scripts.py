import os
import random
import pandas as pd
import json

from tensorflow.keras.models import model_from_json

def model_json_loader(file_path):

    config_file = open(file_path, 'r')
    config_json = config_file.read() 

    return model_from_json(config_json)


def train_validation_test_split(val_percentage=0.15, test_percentage=0.1, trim_ends=45):
    #We get the paths of the images
    images_paths = os.listdir('Data/preprocessed slices')
        
    #We get the codes of the patients (a patient will have 2 images)
    patients_codes = list(set([p.split('_')[2] for p in images_paths])) #Retrieval of patient codes
        
    train_proportion = int(len(patients_codes)*(1-val_percentage-test_percentage))
    validation_proportion = int(len(patients_codes)*val_percentage)
    

    #We partition the codes for training, evaluation and validation
    train_patients = []
    validation_patients = []

    random.shuffle(patients_codes)
    for i in range(train_proportion):
      train_patients.append(patients_codes.pop(-1))

    for i in range(validation_proportion):
      validation_patients.append(patients_codes.pop(-1))
        
    test_patients = patients_codes
    
    #We partition the paths to the file according to the splitting
    #The files of one patient will be both included in the same class of the partition
    train_set = []
    validation_set = []
    test_set = []

    for path in images_paths:
        #we check if we actually add the slice or not
        slice_num = int(path.split('_')[1])
        if ((slice_num < trim_ends) or (slice_num > 176 - trim_ends)):
            continue
      
        for code in train_patients:
            if str(code) in path:
                train_set.append('Data/preprocessed slices/'+path)
                continue
        for code in validation_patients:
            if str(code) in path:
                validation_set.append('Data/preprocessed slices/'+path)
                continue
        for code in test_patients:
                if str(code) in path:
                    test_set.append('Data/preprocessed slices/'+path)
                    continue
    
    #We create the partition for the data
    partition = dict()
    partition['train'] = train_set
    partition['validation'] = validation_set
    partition['test'] = test_set
    
    train_labels = dict()
    for path in partition['train']:
        if '_motion' in path:
            train_labels[path] = 1 #1 -> motion
        else:
            train_labels[path] = 0 #0 -> no motion
    for path in partition['validation']:
        if '_motion' in path:
            train_labels[path] = 1 #1 -> motion
        else:
            train_labels[path] = 0 #0 -> no motion
    
    test_labels = dict()
    for path in partition['test']:
        if '_motion' in path:
            test_labels[path] = 1 #1 -> motion
        else:
            test_labels[path] = 0 #0 -> no motion
            
    #Save the result as a csv partition | path | label
    
    #We save the result in a csv
    arr = []
    for path in partition['train']:
        arr.append(['train', path, train_labels[path]])
    for path in partition['validation']:
        arr.append(['validation', path, train_labels[path]])
    for path in partition['test']:
        arr.append(['test', path, test_labels[path]])
    
    train_col = []
    path_col = []
    label_col = []
    for lst in arr:
        train_col.append(lst[0])
        path_col.append(lst[1])
        label_col.append(lst[2])
        
    df = pd.DataFrame()
    df['partition'] = train_col
    df['path'] = path_col
    df['label'] = label_col
    
    df.to_csv('Data Splits/train_validation_test_split.csv')
    
    return (partition, train_labels, test_labels)

def split_in_k_batches(data, k):
    batch_size = len(data)//k
    data_partition = []
    temp_lst = []
    i = 0
    for j in range(len(data)-1,-1,-1):
        temp_lst.append(data.pop(j))
        i+=1
        if i == batch_size:
            data_partition.append(temp_lst)
            temp_lst = []
            i=0
    #There may be some remaining data points unallocated (at most k-1)
    for i in range(len(data)):
      data_partition[i].append(data.pop(-1))
    return data_partition

def split_in_folds(test_ratio, k):
    df = pd.read_csv('Data Splits/train_validation_test_split.csv', index_col='Unnamed: 0')

    patient_codes = list(set([p.split('_')[2] for p in df['path'].unique()])) #Retrieval of patient codes
    random.shuffle(patient_codes)

    len_test = int((test_ratio)*len(patient_codes))
    
    test_patients = []
    for i in range(len_test):
      test_patients.append(patient_codes.pop(-1))
    
    partition = split_in_k_batches(patient_codes[:], k)
    
    for i in range(len(partition)):
        paths_col = []
        label_col = []
        for path in df['path']:
            for code in partition[i]:
                if code in path:
                    paths_col.append(path)
                    if '_nomotion' in path:
                        label_col.append(0)
                    else:
                        label_col.append(1)
        aux_df = pd.DataFrame()
        aux_df['path'] = paths_col
        aux_df['label'] = label_col
        aux_df.to_csv(f'Data Splits/{i}_fold.csv')
    
    paths_col = []
    label_col = []
    for path in df['path']:
        for code in test_patients:
            if code in path:
                paths_col.append(path)
                if '_nomotion' in path:
                    label_col.append(0)
                else:
                    label_col.append(1)
    test_frame = pd.DataFrame()
    test_frame['path'] = paths_col
    test_frame['label'] = label_col
    test_frame.to_csv(f'Data Splits/test_fold.csv')