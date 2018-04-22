#! /usr/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.data.path.append('/Users/nscsekhar/Desktop/nscsekhar/Desktop/Surya/Personal/MIDS/W266/nltk/nltk')
from comment_parser import comment_parser
from nltk import tokenize
import pandas as pd
import numpy as np
from textblob import TextBlob

import keras
import keras.preprocessing.text as kpt
import matplotlib.pyplot as plt
import sklearn.metrics as sklm

import sys
import os, os.path

def load_dataset(dataset, split = 0.8):
    df = pd.read_csv(dataset)

    row_count = df.shape[0]
    print(row_count)
    #split_point = int(row_count*split)
    #train, test = df[:split_point], df[split_point:]
    train=df.sample(frac=split,random_state=200)
    test=df.drop(train.index)

    train_sentences = train['sentences']
    train_categories = train['readability']

    test_sentences = test['sentences']
    test_categories = test['readability']

    return (train_sentences, train_categories, test_sentences, test_categories)

#
# Routine to extract comments from a given file
#

def has_alphabets(sentence):
    for ch in sentence:
        if ch.isalpha() == True:
            return True
        
    return False

'''
def get_lines_of_code(filename):

    with open(filename) as f:

        fileLineCount = 0
        fileBlankLineCount = 0
        fileCommentLineCount = 0

        for line in f:
            lineCount += 1
            fileLineCount += 1

            lineWithoutWhitespace = line.strip()
            if not lineWithoutWhitespace:
                totalBlankLineCount += 1
                fileBlankLineCount += 1
            elif lineWithoutWhitespace.startswith(commentSymbol):
                totalCommentLineCount += 1
                fileCommentLineCount += 1

        print os.path.basename(fileToCheck) + \
              "\t" + str(fileLineCount) + \
              "\t" + str(fileBlankLineCount) + \
              "\t" + str(fileCommentLineCount) + \
              "\t" + str(fileLineCount - fileBlankLineCount - fileCommentLineCount)
'''        
def get_comment_sents(filename):
    comment_sents = []

    comment_blocks = comment_parser.extract_comments(filename)
    
    #
    # Skip copyright section
    #
    for comment_block in comment_blocks[1:]:
        #
        # Remove any special characters
        #
        comment_text = comment_block._text
        comment_text = comment_text.replace('*', '')
        comment_text = comment_text.replace('\n', '')
        comment_text = comment_text.replace('\t', '')
        comment_text = comment_text.replace('/', ' or ')
        comment_text = comment_text.replace('--', '')
        comment_text = comment_text.replace('i.e.', '')
        
        for sent in tokenize.sent_tokenize(comment_text):
            if has_alphabets(sent) is False:
                continue
            comment_sents.append(sent)

    return comment_sents

# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text, dictionary):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            #print("'%s' not in training corpus; ignoring." %(word))
            pass

    return wordIndices

def plot_model_history(history):

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()    

#
# Routines to extract comments from a given file
#
def has_alphabets(sentence):
    for ch in sentence:
        if ch.isalpha() == True:
            return True
        
    return False

def get_comment_sents(filename):
    comment_sents = []

    comment_blocks = comment_parser.extract_comments(filename)
    
    #
    # Skip copyright section
    #
    for comment_block in comment_blocks[1:]:
        #
        # Remove any special characters
        #
        comment_text = comment_block._text
        comment_text = comment_text.replace('*', '')
        comment_text = comment_text.replace('\n', '')
        comment_text = comment_text.replace('\t', '')
        comment_text = comment_text.replace('/', ' or ')
        
        for sent in tokenize.sent_tokenize(comment_text):
            if has_alphabets(sent) is False:
                continue
                
            comment_sents.append(sent)

    return comment_sents


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.auc.append(sklm.roc_auc_score(targ, score))
        self.confusion.append(sklm.confusion_matrix(targ, predict))
        self.precision.append(sklm.precision_score(targ, predict))
        self.recall.append(sklm.recall_score(targ, predict))
        self.f1s.append(sklm.f1_score(targ, predict))
        self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        return

    def print_all(self):
        print("Precision:", max(self.precision))
        print("Recall:", max(self.recall))
        print("F1:", max(self.f1s))
        