from scipy import signal
import os
import librosa


loc_to = "Respiratory_Sound_Database\\np_arrays"
loc_from = "Respiratory_Sound_Database\\audio_and_txt_files"

oldDir = os.getcwd()

import pandas as pd
import numpy as np

df = pd.read_csv("info_recordings.csv")["base_filename"]
count = 0
print("starting for loop")
for file in df:
    if (not (count+2) %6):
        print(count/df.shape[0])


    count += 1
    os.chdir(oldDir)
    os.chdir(loc_from)
    signal, sr = librosa.load(file,sr=22050)
    print(signal)

    exit(0)
    MFCC = librosa.feature.mfcc(signal, sr=sr, n_fft = 2048, hop_length = 512, n_mfcc=13)
    os.chdir(oldDir)
    os.chdir(loc_to)
    np.save(file[:-3]+"npy", MFCC)
    




