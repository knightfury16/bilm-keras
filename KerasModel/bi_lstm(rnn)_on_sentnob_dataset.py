# -*- coding: utf-8 -*-
"""Bi-LSTM(RNN) on sentNoB dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m_D1C3K64Vu2rovyM9XRlh0ZXKm7U9Yy

# Upload data
"""

# prompt: upload data file

from google.colab import files
uploaded = files.upload()

"""# Essentials Import

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""# Load the data


"""

sent_train = pd.read_csv('/content/Train.csv')
sent_val = pd.read_csv('/content/Val.csv')
sent_test = pd.read_csv('/content/Test.csv')

"""## Read the data"""

print("Training Data:")
print(sent_train.info())

print("Validation Data:")
print(sent_val.info())

print("Test Data:")
print(sent_test.info())

"""# Some visualization of the data"""

pd.set_option('display.max_colwidth', None)
print(sent_train.head())

# prompt: show a distribution of 'Label' and map '0' for neutral, '1' for positive and '2' for negative
# Create the plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 2 subplots

# Training plot
sent_train['Label'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_xticks((0, 1, 2))
ax1.set_xticklabels(('Positive', 'Negative', 'Neutral'))
ax1.set_title('Distribution of Sentiments in Training Dataset')
ax1.set_xlabel('Sentiment')
ax1.set_ylabel('Count')

# Validation plot
sent_val['Label'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_xticks((0, 1, 2))
ax2.set_xticklabels(('Positive', 'Negative', 'Neutral'))
ax2.set_title('Distribution of Sentiments in Validation Dataset')
ax2.set_xlabel('Sentiment')
ax2.set_ylabel('Count')

# Testing plot
sent_test['Label'].value_counts().plot(kind='bar', ax=ax3)
ax3.set_xticks((0, 1, 2))
ax3.set_xticklabels(('Positive', 'Negative', 'Neutral'))
ax3.set_title('Distribution of Sentiments in Testing Dataset')
ax3.set_xlabel('Sentiment')
ax3.set_ylabel('Count')

# Adjust layout and spacing
plt.tight_layout()  # Adjust spacing between subplots
plt.show()

# prompt: show the percentage of Label

sent_train['Label'].value_counts(normalize=True).plot(kind='bar')
plt.xticks((0,1,2),('Positive','Negative','Neutral'))
plt.title('Percentage of Sentiments in Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Percentage')
plt.show()

# prompt: print count of label and total data

total_data = len(sent_train)
neutral_data = sent_train['Label'].value_counts()[0]
positive_data = sent_train['Label'].value_counts()[1]
negative_data = sent_train['Label'].value_counts()[2]



print("Number of datapoints:",total_data)
print("Number of positives:", positive_data, "Percentage:", round((positive_data/total_data)*100),"%")
print("Number of negatives:",negative_data, "Percentage:", round((negative_data/total_data)*100),"%")
print("Number of neutrals:",neutral_data, "Percentage:", round((neutral_data/total_data)*100),"%")

"""## Calculate the average word count and unique word count of the data

Example list for testing
"""

example_data = ["ভাই আপনার কথাই যাদু রয়েছে। , ? | . ! *","ভাই আপনার কথাই","ভাই আপনার কথাই যাদু",]

# print(len(example_data.split()))

import string

def average_word_count(data, show_plots=True):
  total_words = 0
  all_lengths = []
  all_words = []
  for sentence in data:
    words = [word for word in sentence.split() if word not in string.punctuation and not word.isdigit()]
    total_words += len(words)
    all_words.extend(words)  # Add all words for uniqueness analysis
    all_lengths.append(len(words))
  avg_word_count = total_words / len(data)

  # Optionally display distribution of word lengths in each sentence
  if show_plots:
    plt.hist(all_lengths)
    plt.xlabel("Word Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Word Lengths in a Sentence")
    plt.axvline(avg_word_count, color='red', linestyle='dashed', linewidth=2, label=f"Average Word Length ({avg_word_count:.2f})")
    plt.xlim(1, 60)
    plt.grid()
    plt.show()

  unique_words = set(all_words)  # Use a set for efficient uniqueness

  return avg_word_count, len(unique_words)

# Uncomment for testing
# print(average_word_count(example_data, show_plots=False))

"""### Calculating average word for training data"""

avg_word_train, unq_word_train = average_word_count(sent_train["Data"])

print("Average word: ", avg_word_train)
print("Unique words: ", unq_word_train)

"""### Calculating average word for validation data"""

avg_word_val, unq_word_val = average_word_count(sent_val["Data"])

print("Average word: ", avg_word_val)
print("Unique words: ", unq_word_val)

"""# Tokenize the words

### Import from keras
"""

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

"""Instantiate tokenizer"""

tokenizer = Tokenizer(filters='[a-zA-Z0-9!"#$%&()*+,-./:;<=>?@[\]^_`{|।}~ ]+', lower= False)  # filter out english charecters and numbers

"""Fit on text

### For Testing
"""

example_data_2 = ["যাদু রয়েছে"]

example_combined = example_data + example_data_2

print(example_combined)

# tokenizer.fit_on_texts(example_data)

"""# Should not do it, might overfit
## Combine data
"""

# prompt: combine both sent_train['Data'] and sent_val['Data']

combined_data = pd.concat([sent_train['Data'], sent_val['Data']], axis=0)

print(len(combined_data))

"""## Tokenize on only test data"""

tokenizer.fit_on_texts(sent_train['Data'])

# tokenizer.fit_on_texts(sent_train['Data'])
tokenizer.fit_on_texts(combined_data)  # Fit on combined data

word2index = tokenizer.word_index
print(word2index)
print(len(word2index) + 1)

"""## Text to sequence

## For training
"""

X_train_tokens = tokenizer.texts_to_sequences(sent_train['Data'])

print(X_train_tokens[:1])

"""## For validation"""

X_val_tokens = tokenizer.texts_to_sequences(sent_val['Data'])

print(X_val_tokens[:2])

"""## Pad the Xtokens

### Variable for max length
"""

# Setting it manually. Should do it dynamically
max_len = 20

Xtrain = pad_sequences(X_train_tokens, maxlen=max_len, padding = "post", truncating = 'post')
Xval = pad_sequences(X_val_tokens, maxlen=max_len, padding = "post", truncating = 'post')

print(Xtrain)

print(Xval)

"""### Ready the Ytrain and Yval"""

Ytrain = to_categorical(sent_train['Label'])
Yval = to_categorical(sent_val['Label'])

print(Ytrain)
print(Yval)

"""# Create the model

### Import
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SimpleRNN, GRU, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support

"""## Ready the test data"""

# Tokenize the words
Xtokens_test = tokenizer.texts_to_sequences(sent_test['Data'])

# Pad the Xtokens
Xtest = pad_sequences(Xtokens_test, maxlen=max_len, padding = "post", truncating = 'post')

# Ready the Ytest
Ytest = to_categorical(sent_test['Label'])

"""### Model

## A normal model
"""

model = Sequential()

model.add(Embedding(input_dim=len(word2index) + 1, output_dim= 128, input_length=max_len))

model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""## Putting two bidirectional lstm essentially doing the same thing as elmo embedding
### Not really. But this will do for now
"""

# prompt: make me a model using 2 bidirectional lstm and 1 lstm and dense layer

model = Sequential()

model.add(Embedding(input_dim=len(word2index) + 1, output_dim= 100, input_length=max_len))

model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""## Model with Simple RNN"""

# prompt: make me a model using 2 bidirectional lstm and 1 lstm and dense layer

simpleRnnModel = Sequential()

simpleRnnModel.add(Embedding(input_dim=len(word2index) + 1, output_dim= 100, input_length=max_len))

simpleRnnModel.add(Bidirectional(LSTM(128, return_sequences=True)))
simpleRnnModel.add(Dropout(0.2))
simpleRnnModel.add(Bidirectional(LSTM(64, return_sequences=True)))
simpleRnnModel.add(Dropout(0.2))
simpleRnnModel.add(SimpleRNN(32))
simpleRnnModel.add(Dropout(0.2))
simpleRnnModel.add(Dense(units=3, activation='softmax'))

simpleRnnModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
simpleRnnModel.summary()

train_and_test(simpleRnnModel)

"""## Model with 2 lstm input same"""

# prompt: make me a model using 2 bidirectional lstm and 1 lstm and dense layer

sameLSTMModel = Sequential()

sameLSTMModel.add(Embedding(input_dim=len(word2index) + 1, output_dim= 128, input_length=max_len))

sameLSTMModel.add(Bidirectional(LSTM(128, return_sequences=True)))
sameLSTMModel.add(Dropout(0.2))
sameLSTMModel.add(Bidirectional(LSTM(128, return_sequences=True)))
sameLSTMModel.add(Dropout(0.2))
sameLSTMModel.add(LSTM(32))
sameLSTMModel.add(Dropout(0.2))
sameLSTMModel.add(Dense(units=3, activation='softmax'))

sameLSTMModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
sameLSTMModel.summary()

"""## This is the best by far"""

train_and_test(sameLSTMModel)

"""## Direct dense with no middle lstm model"""

DirectDenseModel = Sequential()

DirectDenseModel.add(Embedding(input_dim=len(word2index) + 1, output_dim= 128, input_length=max_len))

DirectDenseModel.add(Bidirectional(LSTM(128, return_sequences=True)))
DirectDenseModel.add(Dropout(0.2))
DirectDenseModel.add(Bidirectional(LSTM(128, return_sequences=False)))
DirectDenseModel.add(Dropout(0.2))
DirectDenseModel.add(BatchNormalization())
DirectDenseModel.add(Dense(units=3, activation='softmax'))

DirectDenseModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
DirectDenseModel.summary()

train_and_test(DirectDenseModel)

"""## Funtion to train and test model"""

# prompt: make a funtion that takes model as argument and train it, test it, and print the perfomance metrics

def train_and_test(model, Myepochs = 10):
  # Instantiate early stopping callback
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)
  # Train the model
  history = model.fit(Xtrain, Ytrain, epochs = Myepochs, validation_data=(Xval, Yval), callbacks=[early_stopping])
  # history = model.fit(Xtrain, Ytrain, epochs=20, batch_size=32, validation_data=(Xval, Yval), verbose=1)

  # Test the model

  # Evaluate the model
  score = model.evaluate(Xval, Yval, verbose=1)
  # Print the performance metrics
  print("Test loss:", score[0])
  print("Test accuracy:", score[1])

  # Make predictions on the test data
  predictions = model.predict(Xtest)

  printPerformanceMetrics(predictions)


def printPerformanceMetrics(predictions):
  predictions_argmax = np.argmax(predictions, axis=1)
  precision, recall, f1, _ = precision_recall_fscore_support(sent_test['Label'], predictions_argmax)
  print("Precision:", precision)
  print("Recall:", recall)
  print("F1 score:", np.mean(f1))
  # print("Support:", support)

"""## Run the model

### Without Validation
"""

model.fit(Xtrain, Ytrain, epochs=100)

"""### Run this with validation"""

# Instantiate early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

"""## Run the elmo model here"""

model.fit(Xtrain, Ytrain, epochs=50, validation_data=(Xval, Yval), callbacks=[early_stopping])

model.fit(Xtrain, Ytrain, epochs=100, validation_data=(Xval, Yval), callbacks=[early_stopping])

# prompt: use argmax on prediction

predictions = np.argmax(predictions, axis=1)
print(predictions)

"""## Elmo"""

accuracy = np.mean(predictions == sent_test['Label'])

print("Accuracy: ", accuracy)

from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(sent_test['Label'], predictions)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", np.mean(f1))
# print("Support:", support)

accuracy = np.mean(predictions == sent_test['Label'])

print("Accuracy: ", accuracy)

# prompt: calculate precisoin, recall and f1

from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(sent_test['Label'], predictions)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", np.mean(f1))
# print("Support:", support)

# prompt: show the metrics in a table format, also include the mean of combined precision,recall and f1

import pandas as pd

metrics = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1 score': f1
})

print(metrics)

print("Mean Precision:", metrics['Precision'].mean())
print("Mean Recall:", metrics['Recall'].mean())
print("Mean F1 score:", metrics['F1 score'].mean())