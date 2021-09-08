print("importing")

import numpy as np
import pandas as pd
import time
from tensorflow.keras.callbacks import ModelCheckpoint, Callback#updating model as go
from tensorflow.keras import backend as K
import tensorflow as tf


loadModel = True

checkpoint_filepath = "model\\temp_save.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val



import myTransformerModel as Transformer
from myDataGenerator import CustomDataGen



df = pd.read_csv('info_recordings.csv')

df = df.sample(frac=1, random_state = 7)

all_ids = list(df["id"])
ids = list()
for i in all_ids:
    if (not i in ids):
        ids.append(i)
        
#experimental values for division (try to get size want)
train_ids = ids[:int(len(ids)*.375)]
val_ids = ids[int(len(ids)*.375):int(len(ids)*.565)]

test_ids = ids[int(len(ids)*.565):]


train_df = df[df["id"].isin(train_ids)]
val_df = df[df["id"].isin(val_ids)]
test_df = df[df["id"].isin(test_ids)]
df = train_df
print("data frames")
print("train")
print(df)
print("val")
print(val_df)
print("test")
print(test_df)




#below doesn't work b/c can memorize demographic info
#this one from stack overflow: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
#df, val_df, test_df = \
#              np.split(df.sample(frac=1, random_state=24601), 
#                       [int(.6*len(df)), int(.8*len(df))])
#df = df.sample(frac=1)
print(df)
directoryX ="Respiratory_Sound_Database\\np_arrays"

locY = "Respiratory_Sound_Database\\patient_diagnosis.csv"

print("creating generator")
generator = CustomDataGen(df, directoryX, locY,3713)
val_generator = CustomDataGen(val_df, directoryX, locY,3713,feedMinMax=True, mins=generator.minVals, maxes=generator.maxVals)
test_generator = CustomDataGen(test_df, directoryX, locY,3713,feedMinMax=True, mins=generator.minVals, maxes=generator.maxVals)



print("dimensions")
generator.get_dimensions()

print("getting items")
X, y = generator.__getitem__(0)

print(X.shape)
print(len(y))

#__ = input("continue (just got X and y, trying to plug into model?\n")

print("creating model")


print("creating transformer model")
myModel = Transformer.model(13+20, 3713, -1, 12, 32)

if (loadModel):
    myModel.load_weights(checkpoint_filepath)

print("compiling model")
myModel.compile(optimizer="Adam", loss = 'categorical_crossentropy', metrics = [tf.keras.metrics.categorical_accuracy, get_f1])

print("model summary")
print(myModel.summary())

#print("trying model with returned __get__item")
#myModel.fit(X, y,  epochs = 1, verbose = 1, batch_size = 4)

#__ = input("continue to use generator directly?\n")


print("fitting model")
#myModel.fit(generator.__getitem__(0)[0], generator.__getitem__(0)[1], epochs=1, verbose=1, batch_size = 16, callbacks = [model_checkpoint_callback])
print("AND 2")

myModel.fit(generator,validation_data =val_generator, epochs=4, verbose=1, batch_size = 16, callbacks = [model_checkpoint_callback])#chose best of 4 based on val

myModel.load_weights(checkpoint_filepath)
#alright, let's see how we did on test set

myModel.evaluate(test_generator, verbose = 1)


