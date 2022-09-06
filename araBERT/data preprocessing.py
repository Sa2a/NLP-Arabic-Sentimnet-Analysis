# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:47:43 2022

@author: river
"""

directory = "D:\DEBI/Uottawa/Data Science Applications/kaggle/astd-v1.0/ASTD-master/data/"
file1 = open(directory+'Tweets.txt', 'r',encoding="utf8")
Lines = file1.readlines()

import pandas as pd

file_train = open(directory+'4class-balanced-train.txt', 'r',encoding="utf8")
trainLines = file_train.readlines()

file_val = open(directory+'4class-balanced-validation.txt', 'r',encoding="utf8")
valLines = file_val.readlines()

file_test = open(directory+'4class-balanced-test.txt', 'r',encoding="utf8")
testLines = file_test.readlines()

l = Lines[int(trainLines[0])].strip().split("\t")
c  = l[1]
print(c =="POS")

balanced = trainLines+valLines+ testLines

tweets = []
target = []
for index in balanced:
    l = Lines[int(index)].strip().split("\t")
    c  = l[1]
    if c == "POS":
        tweets.append(l[0])
        target.append("pos")
    elif c == "NEG":
        tweets.append(l[0])
        target.append("neg")
    elif c == "NEUTRAL":
        tweets.append(l[0])
        target.append("neu")
        
        
balanced_df = pd.DataFrame({"tweet":tweets,"class":target})

print(balanced_df["class"].value_counts())

main_directory = "D:/DEBI/Uottawa/Data Science Applications/kaggle/"

old_data_frame = pd.read_csv(main_directory+"train.csv")

print(old_data_frame["class"].value_counts())

pos = old_data_frame[old_data_frame["class"] == "pos"]
neg = old_data_frame[old_data_frame["class"] == "neg"]
neu = old_data_frame[old_data_frame["class"] == "neu"]
minLen = len(neu.tweet)
merged_df = pd.concat([balanced_df, pos.iloc[:minLen,:],neg.iloc[:minLen,:],neu.iloc[:minLen,:]], axis=0)

print(merged_df["class"].value_counts())

merged_df = merged_df.sample(frac=1,random_state =0)
merged_df.to_csv(main_directory+"araBERT/balanced_data.csv",index= False, encoding="utf8")



import regex as re
import string
import sys
import argparse

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)



# def ara_preprocess(text):!
directory = "D:/DEBI/Uottawa/Data Science Applications/kaggle/"

df = pd.read_csv(directory+"araBERT/balanced_data.csv")
df_test = pd.read_csv(directory+"test.csv")

general_filter = lambda x: re.sub(r'([@A-Za-z0-9ـــــــــــــ]+)|[^\w\s]|#|http\S+', '', x)
_filter = lambda x: re.sub(r'_|[\s]{2,}', ' ', x)
rem_repeated_letters = lambda x: x[0:2] + ''.join([x[i] for i in range(2, len(x)) if x[i]!=x[i-1] or x[i]!=x[i-2]])
heart_emotion_translate = lambda x: re.sub(r'[♥|❤️|❤|♡]+', ' قلب ', x)
happy_emotion_translate = lambda x: re.sub(r'(\^[._]?\^)+', ' سعيد ', x)
sad_emotion_translate = lambda x: re.sub(r'(-[._]?-)+', ' حزين ', x)

data_frame_x = df.copy()
preProcessed = "" 
for x in data_frame_x.tweet:
    x = x.strip("\'")
    x = heart_emotion_translate(x)
    x = happy_emotion_translate(x)
    x = sad_emotion_translate(x)
    x = general_filter(x)
    x = _filter(x)
    x = rem_repeated_letters(x)
    # x = arabert_prep.preprocess(x)
    # x = tokenizer(x).tokens()
    preProcessed+= x+'\n'

import json
import requests


url = 'https://farasa.qcri.org/webapi/spellcheck/'
text = preProcessed 
api_key = "uUtBiqjZKIseAADMMx"
payload = {'text': text, 'api_key': api_key}
data = requests.post(url, data=payload)
result = json.loads(data.text)
print(result) 
    

recorrected_text =result["text"]
exp = recorrected_text[:210]

all_recorrected =  re.findall(r'[\p{Arabic}]+\/[\p{Arabic}_]*|null', recorrected_text)
null_text = re.findall(r'null', recorrected_text)

import numpy as np
a = all_recorrected
labels, counts = np.unique(a,return_counts=True)
import matplotlib.pyplot as plt 
ticks = range(len(counts))
plt.bar(ticks,counts, align='center')
plt.xticks(ticks, labels)

import nltk
freq = nltk.FreqDist(a)
freq.plot(50, cumulative=False)

p = (labels, counts)
correct_text = re.sub(r'\/[\p{Arabic}_]*', '',recorrected_text)

cleaned_tweets = correct_text.split("\n")

new_df = df.copy()
new_df["tweet"] = cleaned_tweets[:-1]

new_df.to_csv(directory+"araBERT/almost_cleaned_data.csv",index = False)









df_test = pd.read_csv(directory+"test.csv")

general_filter = lambda x: re.sub(r'([@A-Za-z0-9ـــــــــــــ]+)|[^\w\s]|#|http\S+', '', x)
_filter = lambda x: re.sub(r'_|[\s]{2,}', ' ', x)
rem_repeated_letters = lambda x: x[0:2] + ''.join([x[i] for i in range(2, len(x)) if x[i]!=x[i-1] or x[i]!=x[i-2]])
heart_emotion_translate = lambda x: re.sub(r'[♥|❤️|❤|♡]+', ' قلب ', x)
happy_emotion_translate = lambda x: re.sub(r'(\^[._]?\^)+', ' سعيد ', x)
sad_emotion_translate = lambda x: re.sub(r'(-[._]?-)+', ' حزين ', x)

data_frame_test = df_test.copy()
preProcessed_tset= "" 
for x in data_frame_test.tweet:
    x = x.strip("\'")
    x = heart_emotion_translate(x)
    x = happy_emotion_translate(x)
    x = sad_emotion_translate(x)
    x = general_filter(x)
    x = _filter(x)
    x = rem_repeated_letters(x)
    # x = arabert_prep.preprocess(x)
    # x = tokenizer(x).tokens()
    preProcessed_tset+= x+'\n'

import json
import requests


url = 'https://farasa.qcri.org/webapi/spellcheck/'
text = preProcessed_tset 
api_key = "uUtBiqjZKIseAADMMx"
payload = {'text': text, 'api_key': api_key}
data = requests.post(url, data=payload)
result = json.loads(data.text)
print(result) 
    

recorrected_text_test =result["text"]

all_recorrected_test =  re.findall(r'[\p{Arabic}]+\/[\p{Arabic}_]*|null', recorrected_text_test)
# null_text = re.findall(r'null', recorrected_text)
correct_text_test = re.sub(r'\/[\p{Arabic}_]*', '',recorrected_text_test)

cleaned_tweets_test = correct_text_test.split("\n")

new_df_test = df_test.copy()
new_df_test["tweet"] = cleaned_tweets_test[:-1]

new_df_test.to_csv(directory+"araBERT/almost_cleaned_test.csv",index = False)


