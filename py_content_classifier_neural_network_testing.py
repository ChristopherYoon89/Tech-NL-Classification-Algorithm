# coding=utf8
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
import time
import numpy as np
import datetime
import json
import csv
import re

# cd /home/chris-yoon/Developing/Freie-Wirtschaft-Content-Classifier

testing_data = pd.read_csv('Training_Data_Content_Classifier_2Classes.csv')
testing_data_sentences = testing_data['Sentence']
#data = data[pd.notnull(data['Text'])]
#data = data[data.Class != 'None']

#print(data)

#train_data = []

#for index,row in data.iterrows():
    #train_data.append({'class':row['Class'], 'sentence':row['Text']})

#print(ignore_words)

#words = []
#classes = []
#documents = []

#with open('Bag_of_Words_Textfile.txt') as f:
    #words = f.read().splitlines()


synapse_file = 'Onlineshop/synapses_language_2classes.json'

with open(synapse_file, encoding='utf8') as data_file:
    synapse = json.load(data_file)
    classes = np.asarray(synapse['classes'])
    words = np.asarray(synapse['words'])
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

#with open(synapse_file, encoding='utf8') as data_file:
    #synapse = json.load(data_file)
    #synapse_0 = np.asarray(synapse['synapse0'])
    #synapse_1 = np.asarray(synapse['synapse1'])

# loop through each sentence in our training data

#for pattern in train_data:
    # tokenize each word in the sentence
    #w = nltk.word_tokenize(pattern['sentence'])
    #print(w)
    # add to our words list
    #words.extend(w)
    # add to documents in our corpus
    #documents.append((w, pattern['class']))
    # add to our classes list
    #if pattern['class'] not in classes:
        #classes.append(pattern['class'])

print(words)

#words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
#words = list(set(words))

# remove duplicates
#classes = list(set(classes))

#print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

# create our training data
#training = []
#output = []
# create an empty array for our output
#output_empty = [0] * len(classes)


def sigmoid(x):
    if np.all(x>=0): #Optimize sigmoid function to avoid extreme data overflow
        output = 1.0 / (1 + np.exp(-x))
        return output
    else:
        output = np.exp(x)/(1+np.exp(x))
        return output

# convert output of sigmoid function to its derivative

def sigmoid_output_to_derivative(output):
    return output*(1-output)



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
                    print ("found in bag: %s" % w)

    return(np.array(bag))



def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values


def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD ]
    results.sort(key = lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]],r[1]] for r in results]
    print ("\n classification: %s" % (return_results))
    return results



#classify('Bürozeiten Home Niebauer Auftraggeber Support Auftragnehmer Support Sachverständigen Leistungen Seminare Schulungen Kontakt Sachverständigen allen Bereichen Komplettanbieter liefern gesamtheitliche Lösungen komplexe Sachverhalte denen Know Erfahrung Bereichen Technik Recht Wirtschaft gefordert UnsÜber Niebauer erbringt Beratungs allen Bereichen Komplettanbieter liefern gesamtheitliche Lösungen komplexe Sachverhalte denen Know Erfahrung Bereichen Technik Recht Wirtschaft gefordert unseren Kunden zählen neben öffentlichen Auftraggebern auch Bauträger Unternehmen Bauhaupt Neben Hilfsgewerbes sowie private Bauherren insb Architekten Zivilingenieure aber auch Rechtsanwälte nach spezifischen bieten Beratungs Support aber auch Umsetzung Leistungen folgenden Themenbereichen Kalkulation Ausschreibung sachverständige Begleitung Vergaben nach ÖNORM Bauabrechnung sowie proaktives aktives Claim Management Projektsteuerung Prozessanalyse Optimierung Seminare Fragen Leistungsangebot Kontaktieren KontaktKontakt Anfahrt Vorgartenstraße')

predicted_values = []

for row in testing_data_sentences:
    try:
        predicted = classify(row)
        print(predicted)
        predicted_values.append(predicted)
    except:
        predicted_values.append('ERROR: Prediction Failed!')

df_predicted_values = pd.DataFrame(predicted_values)


mydata_final = pd.concat([testing_data, df_predicted_values], axis=1)
mydata_final.to_csv('Onlineshop/Predicted_Values_Content.csv', index=False)