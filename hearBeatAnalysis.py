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
