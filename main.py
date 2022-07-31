from cProfile import label
from operator import index
from pyexpat import model
from unittest import result
from urllib import response
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
from pyparsing import Word
from tensorflow.python.framework import ops
# ops.reset_default_graph()
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
with open("intents.json",encoding='utf8') as file:
    data = json.load(file)

# try:
    with open("data.pickle","rb") as f:
        words, labels, traning, output = pickle.load(f)
# except:
    words =[]
    labels =[]
    docs_x =[]
    docs_y=[]

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    traning =[]
    output = []

    out_empty =[0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag =[]

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] =1

        traning.append(bag)
        output.append(output_row)

    traning = numpy.array(traning)
    output = numpy.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, traning, output,),f)

ops.reset_default_graph()
# tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(traning[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    # คอมเม้นปิด model เพื่อฝึกdataใหม่ถ้าเพิ่มข้อมูลเข้ามา
    model.load("model.tflearn")
    # tim.py
except:
    model.fit(traning, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("----------เริ่มคุยกับบอท! (พิมพ์ exit เพื่อหยุดการทำงาน)---------")
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    while True:
        inp = input("| คุณ : ")
        if inp.lower() == "exit":
            print("| บอทคุง : แล้วเจอกันใหม่ (｡╯︵╰｡)")
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag']==tag:
                responses = tg['responses']

        print("| บอทคุง : "+random.choice(responses))

chat()