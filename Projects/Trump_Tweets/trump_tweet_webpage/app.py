# minimal example from:
# http://flask.pocoo.org/docs/quickstart/

from __future__ import print_function
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
#from keras.optimizers import RMSprop
import random
import sys
import re
import tensorflow as tf

df = pd.read_json('trump_12_15_16.json')
df_t = df['text']
ttext = []
for k, v in df_t.iteritems():
    ttext.append(v)
data = ' '.join(ttext)
tweets = re.sub(r"http\S+",'', data.lower(), flags=re.MULTILINE)
tweets = tweets.encode('ascii', errors='ignore')
def clean(txt):
    txt = txt.replace('\n',' ').replace('{','').replace('}','').replace('|','').replace('~','').replace('&amp','').replace('\"','\'')
    txt = txt.replace('\\','').replace('[','').replace(']','').replace('`','').replace('_',' ').replace('/','').replace('+','')
    txt = txt.replace('*','').replace('=','').replace('\r','')
    txt = txt.replace('SPEECH 1','').replace('SPEECH 2','').replace('SPEECH 3','').replace('SPEECH 4','').replace('SPEECH 5','')
    txt = txt.replace('SPEECH 6','').replace('SPEECH 7','').replace('SPEECH 8','').replace('SPEECH 9','').replace('SPEECH 10','')
    txt = txt.replace('    ',' ').replace('   ',' ').replace('  ',' ')
    return txt
tweets = clean(tweets)

with open('speeches.txt', 'r') as myfile:
    speech=myfile.read().strip().decode('utf-8')
speech = speech.encode('ascii', errors='ignore')
speech_list = speech.split("SPEECH ")
speech_doc = []
for i in speech_list:
    speech_doc.append(clean(i))
speech_doc.pop(0)
doc10 = speech_doc.pop(9) + speech_doc.pop(9)
speech_doc_1 = []
count = 0
for d in speech_doc:
    count += 1
    if count == 10:
        speech_doc_1.append(d[2:])
    else:
        speech_doc_1.append(d[1:])
speech = clean(speech)
speech = speech.lower()

text = clean(tweets + speech)

chars = sorted(list(set(text)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 52
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))

filename = "weights-improvement-00-1.0309.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

graph = tf.get_default_graph()

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

losers  = ["Republicans must stop relying on losers like @KarlRo",
           "@realDonaldTrump The biggest loser in the debate was",
           "DonaldTrump was right about #RosieODonnell. She is a",
           "Despite what the haters and losers like to say, I ne"]

dumbest = ["Bryant Gumbel is one of the dumbest racists around -",
           "Really stupid leadership that wants Al Sharpton to h",
           "after all these years of watching stupidity, we will",
           "I loved beating these two terrible human beings. I w",
           "Mitt Romney is one of the dumbest and worst candidat"]
           
mediadis =["Megyn Kelly must have had a terrible vacation, she i",
           "If crazy @megynkelly didn't cover me so much on her ",
           "Jon Stewart is a joke, not very bright and totally o",
           "@megynkelly recently said that she can't be wooed by",
           "@NYDailyNews but little Mort Zuckerman is franticall"]

obama =   ["We will immediately repeal and replace ObamaCare - a",
           "President Obama is the worst president in U.S. histo",
           "Obama has done such a poor job as president, you won",
           "Why is Obama playing basketball today? That is why o",
           "President Obama was terrible on @60Minutes tonight. ",
           "ObamaCare is one of the worst political disasters of",
           "Obama is truly a pathetic excuse of a president, can"]

hillary = ["Bill Clinton is the WORST abuser of woman in U.S. po",
           "Hillary Clinton is not qualified to be president bec",
           "If Hillary thinks she can unleash her husband, with ",
           "Is @realDonaldTrump debating Crooked @HillaryClinton",
           "Crooked Hillary has no clue! @Teamsters I am going t"]

great =   ["I am the only one who can Make America Great Again! ",
           "Thank you Florida - we are going to MAKE AMERICA GRE",
           "It is time to take back our country- and MAKE AMERIC",
           "@CindyBl: I voted 4 Reagan to make America Great. I "]

real =    ["@realDonaldTrump The biggest loser in the debate was",
           "I'm worth $8 billion and you're worth peanuts...with",
           "@realDonaldTrump is going to cut taxes BIG LEAGUE -C",
           "Is @realDonaldTrump debating Crooked @HillaryClinton"]

random_topic = losers + mediadis + obama + hillary + dumbest + great + real

total_list = [random_topic, losers, mediadis, obama, hillary, dumbest, great, real]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/score", methods=["GET", "POST"])
def score():   
    parameters = request.json

    diversity = float(parameters['v'][0])
    length = int(parameters['v'][1])
    topic = int(parameters['v'][2])
    
    global graph
    with graph.as_default(): 
        generated = ''
        sentence = random.choice(total_list[topic]).lower()
        generated += sentence

        for i in range(length):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
        #print(generated)
    try:
        generated = '.'.join(map((lambda x: x[0].upper()+x[1:]), generated.replace('. ','.').split('.'))).replace('.', '. ')
    except:
        pass
    reply = {'tweet': generated}

    return jsonify(reply)

# Start the app server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
