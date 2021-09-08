import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import pandas as pd
import numpy as np
import librosa
import os

from tensorflow.keras.utils import to_categorical

#have seperate file w/ dataframe of info, will make easier (can have just df instead of mult directories and idList and such)
from random import randint
#based on instruction from https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, df_, directoryX, locY, maxLength=False,feedMinMax = False, mins = None, maxes = None,regularize = True, isAudio = False, batch_size=8):
        self.locX = directoryX
        self.locY = locY#should be path and file
        self.df = df_
        self.batch_size = batch_size

        self.isAudio = isAudio
        
        if (not maxLength):#we'll discover maxLength for ourselves
            filenames = self.df[self.df.columns[-1]]
            oldDir = os.getcwd()
            os.chdir(directoryX)
            maxLength = 0 
            for name in filenames:
                MFCC = self.process_video(name)
                l = MFCC.shape[1]
                if (l>maxLength):
                    maxLength = l
            
                
            os.chdir(oldDir)
        self.maxLength = maxLength
            
            
        
        self.index_max = self.df.shape[0]
        validVideos = os.listdir(self.locX)
        validVideos = [i for i in validVideos if i[-3:]=='wav']
        self.validVideos = validVideos

        if (regularize):
            df_temp = df_[df_.columns[2:-1]]
            if (not feedMinMax):
                maxes = df_temp.max(axis=0)
            
            self.maxVals = maxes
            if (not feedMinMax):
                mins = df_temp.min(axis=0)
            self.minVals = mins
            df_temp = (df_temp-mins)/(maxes-mins)
            for col in df_temp.columns:
                self.df[col] = df_temp[col]
            

        self.set_y(locY)

    def process_video(self, filename):#assumes in correct directory
        if (self.isAudio):
            signal, sr = librosa.load(filename,sr=22050)
            MFCC = librosa.feature.mfcc(signal, sr=sr, n_fft = 2048, hop_length = 512, n_mfcc=13)
        else:
            savedFile = filename[:-3]+"npy"
            MFCC = np.load(savedFile)
            
            #filename is an np array
            

        return MFCC
        

    def my_padding(self, MFCC):#MFCC is k by t

        excess = np.zeros((MFCC.shape[0], self.maxLength-MFCC.shape[1]))-1

        MFCC = np.concatenate((MFCC,excess ), axis = 1)
      
        return MFCC
    def __get_input(self,df):
        demographic_info = np.array(df[df.columns[2:-1]])#not passing id and not passing filename
        demographic_info = demographic_info[..., np.newaxis]#m by k by 1
        filenames = df['base_filename']
        old_dir = os.getcwd()
        os.chdir(self.locX)
        MFCCs = list()#m by k2 by t
    
        the_keys = filenames.index
        for num in range(len(filenames)):
            filename = filenames[the_keys[num]]
            #signal, sr = librosa.load(filename,sr=22050)
            #MFCC = librosa.feature.mfcc(signal, sr=sr, n_fft = 2048, hop_length = 512, n_mfcc=13)
            MFCC = self.process_video(filename)

            MFCC = self.my_padding(MFCC)
            MFCCs.append(MFCC)
            #of form shape by numTimes
        #MFCCs = tf.keras.preprocessing.sequence.pad_sequences(MFCCs, maxlen = self.maxLength)
        MFCCs = np.array(MFCCs)

        reshaped_dem = np.broadcast_to(demographic_info, (MFCCs.shape[0], demographic_info.shape[1], MFCCs.shape[2]))


        X = np.concatenate((MFCCs,reshaped_dem ), axis = 1)

        
        os.chdir(old_dir)

        return X
        

            
        


        """
sample_rate, samples = wavfile.read('Respiratory_Sound_Database\\audio_and_txt_files\\101_1b1_Al_sc_Meditron16BIT.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)


        """
        
    def set_y(self, file):
        f = open(file, 'r')
        l = f.read().split("\n")
        f.close()
        dictionary = {float(k.split(',')[0]):k.split(',')[1] for k in l}
        vals = list()
        for i in dictionary.values():
            if i in vals:
                continue
            vals.append(i)
        numCats = len(vals)
        self.numY = numCats

        self.id_to_disease = dictionary

        self.disease_to_one_hot =  {vals[i]: to_categorical(i, numCats) for i in range(numCats)}
        #list.index(max(list)) will give index (two passes but list is short)

        d = self.disease_to_one_hot

        self.id_to_one_hot = {key: self.disease_to_one_hot[self.id_to_disease[key]] for key in self.id_to_disease}
        self.place_to_disease = {d[key].argmax(): key for key in d.keys()}
        
        
        
        
        
        
        
    def __getitem__(self, index):#returns X,y pairs
        #print("starting __getitem__")
        relevant = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__get_input(relevant)
        y = [self.id_to_one_hot[this_id] for this_id in relevant['id']]

        #print(X[1, 2,:])
        X = np.transpose(X, axes=(0,2,1))
        #print(X[1,:,2])
        #__ = input("check above are the same")
        #print("Returning X, y pair")

        self.currentY = np.array(y)

        print("the y data")
        print(self.currentY.shape)
        print(self.currentY)
        return X,np.array(y) 
        
        

    def on_epoch_end(self, ):
        #print("epoch end, shuffling dataframe")
        print("the y data")
        print(self.currentY.shape)
        print(self.currentY)
        self.df = self.df.sample(frac=1)

    def __len__(self,):
        return (int(np.floor(self.index_max/self.batch_size)))

    def get_dimensions(self):#rlly print dimensions, but just needed to see value not get it
        print("X dimensions (0-2 as examples)")
        print(self.__get_input(self.df.iloc[0:2]).shape)
        print("y dimensions")
        print(self.numY)
    
        
    
