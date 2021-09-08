print("importing")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import soundfile
import os

#from https://stackoverflow.com/questions/44812553/how-to-convert-a-24-bit-wav-file-to-16-or-32-bit-files-in-python3
def convertAllFilesInDirectoryTo16Bit(directory):
    for file in os.listdir(directory):
         if(file.endswith('.wav')):
             nameSolo = file.rsplit('.', 1)[0]
             print(directory + nameSolo )
             data, samplerate = soundfile.read(directory + file)                
            
             if (not nameSolo.find('16BIT')+1):#added this part too
                 soundfile.write(directory + nameSolo + '16BIT.wav', data, samplerate, subtype='PCM_16')
                 print("converting " + file + "to 16 - bit")
                 os.remove(directory+file)#I added this part

if (False):
    convertAllFilesInDirectoryTo16Bit('Respiratory_Sound_Database\\audio_and_txt_files\\')
#from https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3


#below is based on tutorial from https://www.youtube.com/watch?v=Oa_d-zaUti8
import librosa, librosa.display
import matplotlib.pyplot as plt

signal, sr = librosa.load("Respiratory_Sound_Database\\audio_and_txt_files\\101_1b1_Al_sc_Meditron16BIT.wav",sr=22050)
#signal 1D np array is sr * T (seperation rate * 2) = (22050 * 30)

#librosa.display.waveplot(signal, sr=sr)#sr is 22050 (input)
#code to plot is below
#plt.xlabel("Time")
#plt.ylabel("Amplitude")
#plt.show()

#now frequency
import numpy as np

fft = np.fft.fft(signal)#np array of size total num samples (sr * T)
#it's complex, get magnitude

magnitude = np.abs(fft)#indicate contribution of each frequneyc to sound
#magnitude has magnitude of each frequency beam
frequency = np.linspace(0, sr, len(magnitude))#gives number of evenly spaced numbers in interval
#frequency is like the corresponding frequency

#plotting
"""
plt.plot(frequency, magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
"""

#notice that aside from magnitude[k] = magnitude[-k], meaning we don't need all info
#thus, we only need left frequency

left_frequency = frequency[:int(len(frequency)/2+1)]
left_magnitude = magnitude[:int(len(frequency)/2+1)]
"""
plt.plot(left_frequency, left_magnitude)

plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
"""

#stft to get spectogram

n_fft = 2048 #number of samples per fft
#above is the window considering when performing single fourier transform (b/c we repeat across time)

hop_length= 512 #about we shift each fourier transform to right

#2048 are commonlyk used for speech and music
stft = librosa.core.stft(signal, hop_length = hop_length, n_fft = n_fft)

spectogram = np.abs(stft)#complex -> magnitude

#plotting result

#heat map like data
"""
librosa.display.specshow(spectogram, sr=sr, hop_length = hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()#for how amplitude varies through spectogram
plt.show()
"""

#notice that our visualization is linear (not how we hear sound) so we should use log spectogram instead
"""
log_spectogram = librosa.amplitude_to_db(spectogram)

librosa.display.specshow(log_spectogram, sr=sr, hop_length = hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()#for how amplitude varies through spectogram
plt.show()
"""


#extracting mfccs

#n_mfcc is number of coefficients to extract (13 common for music)
MFCCs = librosa.feature.mfcc(signal, sr= sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
print(MFCCs.shape)
librosa.display.specshow(MFCCs, sr = sr, hop_length = hop_length)
plt.xlabel("TIME")
plt.ylabel("MFCC")
#y axis, each is a coefficient
plt.colorbar()
plt.show()

#mfcc should be used for deep learning
#alternate can be log Mel filterbank features (fbanks)


        
