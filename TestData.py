

import os
import sys
from pathlib import Path
from Testdata_Filecreation import preprocess
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sckleatn.metrics import f1_score


sys.path.append(Path(os.path.join(os.path.abspath(''), '../')).resolve().as_posix())



import json
import pandas as pd
import re
import nltk
from time import time
from emoji import demojize



def csvTestFileCreation():
    file = open("nlp_test.json")
    data = json.load(file)
    df = pd.DataFrame(columns=["id","text"])
    for key in data:
        df = df.append({"id":key,"text":data[key]["body"]}, ignore_index = True)
    df.to_csv("nlp_test.csv")


def preprocess(texts, quiet=False):
  start = time()
  # Lowercasing
  texts = texts.str.lower()

  # Remove special chars
  texts = texts.str.replace(r"(http|@)\S+", "")
  texts = texts.apply(demojize)
  texts = texts.str.replace(r"::", ": :")
  texts = texts.str.replace(r"’", "'")
  texts = texts.str.replace(r"[^a-z\':_]", " ")

  # Remove repetitions
  pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
  texts = texts.str.replace(pattern, r"\1")

  # Transform short negation form
  texts = texts.str.replace(r"(can't|cannot)", 'can not')
  texts = texts.str.replace(r"n't", ' not')

  # Remove stop words
  stopwords = nltk.corpus.stopwords.words('english')
  stopwords.remove('not')
  stopwords.remove('nor')
  stopwords.remove('no')
  texts = texts.apply(
    lambda x: ' '.join([word for word in x.split() if word not in stopwords])
  )

  if not quiet:
    print("Time to clean up: {:.2f} sec".format(time() - start))

  return texts




csvTestFileCreation()




tokenizer_path = Path('tokenizer.pickle').resolve()
with tokenizer_path.open('rb') as file:
    tokenizer = pickle.load(file)




input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
num_classes = 12
embedding_dim = 750
input_length = 150
lstm_units = 130
lstm_dropout = 0.1
recurrent_dropout = 0.1
spatial_dropout=0.2
filters=64
kernel_size=3




input_layer = Input(shape=(input_length,))
output_layer = Embedding(
  input_dim=input_dim,
  output_dim=embedding_dim,
  input_shape=(input_length,)
)(input_layer)

output_layer = SpatialDropout1D(spatial_dropout)(output_layer)

output_layer = Bidirectional(
LSTM(lstm_units, return_sequences=True,
     dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
)(output_layer)
output_layer = Conv1D(filters, kernel_size=kernel_size, padding='valid',
                    kernel_initializer='glorot_uniform')(output_layer)

avg_pool = GlobalAveragePooling1D()(output_layer)
max_pool = GlobalMaxPooling1D()(output_layer)
output_layer = concatenate([avg_pool, max_pool])

output_layer = Dense(num_classes, activation='softmax')(output_layer)

model = Model(input_layer, output_layer)




model_weights_path = Path('model_weights.h5').resolve()
model.load_weights(model_weights_path.as_posix())



data_path = Path('nlp_test.csv').resolve()
data = pd.read_csv(data_path)




encoder_path = Path('encoder.pickle').resolve()
with encoder_path.open('rb') as file:
    encoder = pickle.load(file)


cleaned_data = preprocess(data.text)
sequences = [text.split() for text in cleaned_data]
list_tokenized = tokenizer.texts_to_sequences(sequences)
x_data = pad_sequences(list_tokenized, maxlen=150)



y_pred = model.predict(x_data)
label = encoder.classes_
emotions = ""
for post_val in range(0, len(y_pred)):
    post_avg = np.average(y_pred[post_val])
    post_text = data.iloc[post_val]["text"]
    
    for index in range(0, len(y_pred[post_val])):
        if(y_pred[post_val][index] > post_avg):
            emotions += str(label[index]) + "; "
    print(data.iloc[post_val]["id"] + " : " + emotions[:-2])
    emotions = ""
           
    print("\n")



f1_score()




