import sys
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_yaml

import pandas as pd
import numpy as np
import json

import common

from importlib import reload

print("File name:", sys.argv[1])
maxlen = 400
#
# Load the model
#
yaml_file = open("models/cnn_model.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()

loaded_model = model_from_yaml(loaded_model_yaml)

# load weights into new model
loaded_model.load_weights("models/cnn_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#
# Get comment sentences
#
comment_sents = []
comment_sents += common.get_comment_sents(sys.argv[1])
# print(len(comment_sents))
# for sent in comment_sents:
#     print(sent)

#
# Predict the comment sentences
#
reload(common)
pred_tokenizer = Tokenizer(num_words=20000)
labels = ['Good Comment', 'Needs Fix']

with open('models/dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

for sent in comment_sents:
    print("Sentence: \n",sent)
    comment_sequence = common.convert_text_to_index_array(sent, dictionary)
    comment_vector = sequence.pad_sequences([comment_sequence], maxlen=maxlen)

    pred = loaded_model.predict_classes(comment_vector)
    print("Readability score: ", labels[int(pred)])

    #print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
