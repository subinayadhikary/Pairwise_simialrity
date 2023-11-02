import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import math
from nltk.stem import PorterStemmer
import nltk
import re
import numpy as np
from numpy.linalg import norm
import nltk
import json
stopwords=[]
ps = PorterStemmer()
vector={}
from nltk.stem.wordnet import WordNetLemmatizer
json_value_idf=open("/home/subinay/Documents/data/pairwise_similarity/tf_idf/idf_value_corpus.json")
Idf_value= json.load(json_value_idf)
dir_annotated="/home/subinay/Documents/data/pairwise_similarity/K-fold/200_docs/"
def top_terms(score,term):  # to find embedding of top tf-idf terms
    feature_scores = {}
    for i in range(len(score)):
        feature_scores[term[i]]=score[i]*Idf_value[term[i]]
    term_score=dict(sorted(feature_scores.items(), key=lambda item: item[1],reverse=True))
    docs1=[]
    for key,value in term_score.items():
        if(value!=0):
            docs1.append(key)
    num_to_select = int(1* len(docs1)) # to select top terms based on their tf-idf value
    # select the first 10% of the elements
    selected_elems = docs1[:num_to_select]
    tfidf_value={}
    for elem in selected_elems:
        tfidf_value[elem]=feature_scores[elem]
    return tfidf_value          
docs=[]
docid=[]
files=os.listdir(dir_annotated)
for file in files: 
    docid.append(file)
    text=""
    with open("/home/subinay/Documents/data/pairwise_similarity/K-fold/200_docs/"+file,"r") as f1:
        for line in f1:
            line=line.strip()
            text=text+line
    docs.append(text)
with open("/home/subinay/Documents/data/pairwise_similarity/stopword/stopword_5000_10.txt","r") as f1:
    for line in f1:
        line=line.strip()
        stopwords.append(line)
vectorizer = CountVectorizer()
vectorizer.set_params(stop_words=stopwords)
stemmer = PorterStemmer()
analyzer = vectorizer.build_analyzer()
vectorizer.set_params(tokenizer=lambda doc: (stemmer.stem(token) for token in analyzer(doc)))
X = vectorizer.fit_transform(docs)
# Access the vocabulary and feature names
vocabulary = vectorizer.vocabulary_
term= vectorizer.get_feature_names()
X1=[]
score=[]
count=0
for i in range(0,200):
    count=count+1
    x=X[i,].toarray()
    x1=x.flatten()
    score.append(x1)
for i in range(len(score)):
    text=top_terms(score[i],term)
    X1.append(text)
for i in range(len(docid)):
    vector[docid[i]]=X1[i]
with open("tf_idf_embedding_200_docs_1.json", "w") as outfile:
     json.dump(vector, outfile)
