Alright, for this projecct I used the dataset from: https://www.kaggle.com/vbookshelf/respiratory-sound-database

Pretty much, it contains demographic information, instrument information, and the heartbeat of patients along with their diagnosis. Using the encoder section of the Transformer model, MICE, and general deep learning techniques, I analyzed the heartbeat and tried to make predictions.
Note that the train/val/test sets were divided based on patient so the demographic information couldn't be memorized.

Alright, let's go file by file

<h1>createDF.py</h1>
<li>used MICE implementation to fill out missing information (also marked with 0 or 1 in seperate column)        </li>      
<li>created a csv file used in the dataframe</li>

<h1> heartBeatAnalysis.py</h1>
<li> converts all wav files to 16 bit so I can process them </li>

<h1> myDataGenerator.py </h1>
<li> employed padding and regularization</li>
<li> accepted both processed (np array) and unprocessed (wav) files based on what was passed into the dataframe</li>

<h1> myModel.py </h1>
<li> utilized model-checkpoints to save model on val accuracy (also callbacks were used to analyze F1-score) </li>
<li> split sets into train/val/test based on user id so model couldn't memorize user information </li>

<h1> myTransformer.py </h1>
<li> implements the encoder part of the state of the art Transformer model</li>
<li> this file also includes the rest of the model </li>

<h1> saveNPArrays.py </h1>
<li> casts WAV -> NP for files (uses librosa for data processing) allowing trade-off between space of data and time for processing </li>
