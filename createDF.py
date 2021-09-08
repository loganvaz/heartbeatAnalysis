#MICE implementation from https://www.geeksforgeeks.org/missing-data-imputation-with-fancyimpute/
import numpy as np
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from keras.utils.np_utils import to_categorical



import pandas as pd

d = open("demographic_info.txt")
demographics = d.readlines()
d.close()



d = {}
for line in demographics:
    if (line.strip()==""):
        print("null line")
        continue
    arr = line.strip().split(" ")
    id_ = float(arr[0])
    certainty = list()
    try:
        age = float(arr[1])
        certainty.append(1)
    except:
        age = float(arr[1]+"N")#N -> NAN, not just NA
        certainty.append(0)
    sex = 1*(arr[2]=="F")
    try:
        bmi =float( arr[3])
        certainty.append(1)
    except:
        bmi = float(arr[3]+"N")
        certainty.append(0)
    try:
        child_weight = float(arr[4])
        certainty.append(1)
    except:
        child_weight = float(arr[4]+"N")
        certainty.append(0)
    try:
        adult_weight = float(arr[5])
        certainty.append(1)
    except:
        adult_weight = float(arr[5]+"N")
        certainty.append(0)
    
    
    d[id_] = [id_, age, sex, bmi, child_weight, adult_weight]+certainty

import os

os.chdir("Respiratory_Sound_Database")
os.chdir("audio_and_txt_files")

filenames = os.listdir()
the_filenames_wav = [i for i in filenames if i[-3:]== 'wav']
filenames = [i[:-9].split("_") for i in filenames if i[-3:]=='wav']
os.chdir("..")
os.chdir("..")
def process(entry):
    t = (entry.find("Tc")!=-1)*1
    A = (entry.find("A")!=-1)*1
    P = (entry.find("P")!=-1)*1
    L = (entry.find("L")!=-1)*1
    l = (entry.find("l")!=-1)*1
    r = (entry.find("r")!=-1)*1
    return [t,A,P,L,l,r]
    
instrument = {}

info_filenames = list()
in_order_filenames = list()

for entry in filenames:
    id_, location_one_hot, acquisition  = int(entry[0]), process(entry[2]), (entry[3]=='sc')*1
    try:
        measure = instrument[entry[-1]]
    except:
        instrument[entry[-1]] = measure = len(instrument.keys())

    returns = [entry[-1]] + d[id_]
    in_order_filenames.append(entry[-1])
    #filenames, id_info, filename info
    measure = to_categorical(measure, num_classes=4)
    returns += location_one_hot+[acquisition] + list(measure)

    info_filenames.append(returns[1:])

from pandas import DataFrame

cols = ['id', 'age','sex', 'bmi', 'child_weight', 'adult_weight','age_certainty', 'bmi_certainty', 'child_weight_certainty', 'adult_weight_certainty']
cols += ['Tc', 'A', 'P', 'L', 'l', 'r', 'sc', 'measure_device1','measure_device2','measure_device3','measure_device4']
print(len(filenames))
print(info_filenames[0])
df = DataFrame(info_filenames, columns = cols)
print(df)


#okay, we have {} for each person, now we need to add for each file



mice_imputer = IterativeImputer()
#transform will impute all wissing wihout fitting, fit_transform fits and returns X

mice_data = mice_imputer.get_params()
#.set_params(**params)
df =pd.DataFrame( mice_imputer.fit_transform(df), columns = cols)

df["base_filename"] = the_filenames_wav


df.to_csv("info_recordings.csv")


