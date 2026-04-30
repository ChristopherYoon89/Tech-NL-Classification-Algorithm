import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
import time
import numpy as np
import datetime
import json
import csv

nltk.download('punkt_tab')


data = pd.read_excel('Data/Dataset_Final_30042026.xlsx')
data = data[pd.notnull(data['Content'])]
data = data[pd.notnull(data['Subcategory'])]   # <-- add this
data = data[data.Subcategory != 'None']

print(data)

train_data = []

for index,row in data.iterrows():
    train_data.append({'class':row['Subcategory'], 'sentence':row['Content']})


words = []
classes = []
documents = []

stemmer = LancasterStemmer()

for pattern in train_data:
    w = nltk.word_tokenize(pattern['sentence'])
    w = [stemmer.stem(word.lower()) for word in w]

    words.extend(w)
    documents.append((w, pattern['class']))

    if pattern['class'] not in classes:
        classes.append(pattern['class'])


words = [w for w in words if w.isalpha()]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))


print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)

training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [word.lower() for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

print("# words", len(words))
print("# classes", len(classes))


def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative

def sigmoid_output_to_derivative(output):
    return output*(1-output)


def relu_activate(x):
    output = np.maximum(0, x)
    return output 


def relu_output_to_derivative(output):
    return np.where(output > 0, 1, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))


def train(X, y, hidden_neurons=64, alpha=0.1, epochs=20000):
    '''
    Function for training the model
    '''
    print(f"Training with {hidden_neurons} neurons, alpha:{alpha}")

    np.random.seed(1)

    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    for j in range(epochs):

        # Forward pass
        layer_0 = X
        layer_1 = relu_activate(np.dot(layer_0, synapse_0))
        layer_2 = softmax(np.dot(layer_1, synapse_1))

        # Error
        layer_2_error = y - layer_2

        # Backprop (correct for softmax + cross-entropy)
        layer_2_delta = layer_2_error

        layer_1_error = layer_2_delta.dot(synapse_1.T)
        layer_1_delta = layer_1_error * relu_output_to_derivative(layer_1)

        # Update weights
        synapse_1 += alpha * layer_1.T.dot(layer_2_delta)
        synapse_0 += alpha * layer_0.T.dot(layer_1_delta)

        if j % 2000 == 0:
            print(f"Iteration {j} - Error: {np.mean(np.abs(layer_2_error))}")

    # Save model
    now = datetime.datetime.now()

    model = {
        'synapse0': synapse_0.tolist(),
        'synapse1': synapse_1.tolist(),
        'words': words,
        'classes': classes,
        'datetime': now.strftime("%Y-%m-%d %H:%M")
    }

    with open("synapses_model.json", "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)

    print("Model saved.")


X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=64, alpha=0.1, epochs=20000)

elapsed_time = time.time() - start_time
print("processing time:", elapsed_time, "seconds")