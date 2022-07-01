import os
import random

def train_validation_test_split()->(dict,dict,dict):
    #We get the paths of the images
    images_paths = os.listdir('../Data/preprocessed images')
        
    #We get the codes of the patients (a patient will have 2 images)
    patients_codes = set()
    for path in images_paths:
        patients_codes.add(path[:5])
        
    #80% training data, 15% validation data, 5% testing data
    train_proportion = len(patients_codes)*80//100
    validation_proportion = len(patients_codes)*15//100
    
    #We split the patients randomly
    train_patients = random.sample(patients_codes, train_proportion)
    
    for code in train_patients:
       patients_codes.remove(code)
       
    validation_patients = random.sample(patients_codes, validation_proportion)
    for code in validation_patients:
        patients_codes.remove(code)
        
    test_patients = list(patients_codes)
    
    ##try shuffle
    
    #We partition the paths to the file according to the splitting
    #The files of one patient will be both included in the same class of the partition
    train_set = []
    validation_set = []
    test_set = []

    for path in images_paths:
        for code in train_patients:
            if str(code) in path:
                train_set.append(path)
                continue
        for code in validation_patients:
            if str(code) in path:
                validation_set.append(path)
                continue
        for code in test_patients:
                if str(code) in path:
                    test_set.append(path)
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
    #save the result as a csv partition | path | label
    return (partition, train_labels, test_labels)