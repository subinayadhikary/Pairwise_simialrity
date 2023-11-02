import os
import json
import re
import numpy as np
from numpy.linalg import norm
import math
from collections import Counter
from sklearn.metrics import jaccard_score
NN1=[]
NN2=[]
## annotated information of 200 documents 
json_data=open("/home/subinay/Documents/data/sentence_tag/search_file_new_2.json") 
jdata = json.load(json_data)
json_embedding_tf_idf=open("/home/subinay/Documents/data/pairwise_similarity/K-fold/tf_idf_10000_200/tf_idf_embedding_200docs.json")
embedding= json.load(json_embedding_tf_idf)
json_embedding_section=open("/home/subinay/Documents/data/pairwise_similarity/supervised_non_prarmetric/section_embedding_200docs.json")
embedding_section= json.load(json_embedding_section)
dir_train="/home/subinay/Documents/data/pairwise_similarity/First_50/150_docs/"
train_docs=os.listdir(dir_train)
dir_test="/home/subinay/Documents/data/pairwise_similarity/First_50/50_docs/"
test_docs=os.listdir(dir_test)
WORD = re.compile(r"\w+")
def compute_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return round((numerator) / denominator,2)
    
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def sent_score(list1,list2,i,j):
    score=0
    text1 = list1[i][0]  # content from anno1 file
    text2 = list2[j][0]   # content from anno2 file
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2) 
    sim_score=compute_cosine(vector1, vector2)
    annotated_score=math.exp(-(1-sim_score))
    vector1 = text_to_vector(list1[i][2]) # tag(s) for content
    vector2 = text_to_vector(list2[j][2]) # tag(s) for content 
    tag_score=compute_cosine(vector1,vector2)
    score=float(annotated_score*tag_score)  # Taken similarity score of contents and labeled tag
    return score

def doc_sim(docid1,docid2): # compute similarity b/w two documents using cosine similarity
    score,score_tf_idf,score_section=0,0,0
    doc1_embedding=embedding[docid1]
    doc2_embedding=embedding[docid2]
    # A=np.array(doc1_embedding)
    # B=np.array(doc2_embedding)
    # score_tf_idf= np.dot(A,B)/(norm(A)*norm(B))
    score_tf_idf=compute_cosine(doc1_embedding,doc2_embedding)
    doc1_embedding=embedding_section[docid1]
    doc2_embedding=embedding_section[docid2]
    score_section=jaccard_score(doc1_embedding,doc2_embedding)
    score=0.4*float(score_tf_idf)+0.6*float(score_section)
    return score 
# def doc_sim(docid1,docid2):
#     score=0
#     doc1_embedding=embedding[docid1]
#     doc2_embedding=embedding[docid2]
#     score=jaccard_score(doc1_embedding,doc2_embedding)
#     return score


def sim_nn_docs(did1,did2): ## compute similarity b/w two Nearest neighbours documents
    #print(did1,did2)
    list1=jdata[did1]["anno1"]  # used highlighted sentences by anno1
    list2=jdata[did2]["anno1"]  # used heighlighted sentences by anno1
    SimScore,score_anno1,score_anno2=0,0,0
    for i in range(len(list1)):
        sentence_score=[]
        for j in range(len(list2)):
            sentence_score.append(sent_score(list1,list2,i,j))
        score_anno1=score_anno1+max(sentence_score,default=0)
    for i in range(len(list2)):
        sentence_score=[]
        for j in range(len(list1)):
            sentence_score.append(sent_score(list2,list1,i,j))
        score_anno2=score_anno1+max(sentence_score,default=0)
    try:
        score_anno1=float(score_anno1/len(list1))
        score_anno2=float(score_anno2/len(list2))
    except:
        score_anno1=0
        score_anno2=0
    SimScore=(score_anno1+score_anno1)/2
    return SimScore

def document_sim(did1,did2):
    score_doc1={}
    score_doc2={}
    similarity_score,similarity_score_1,similarity_score_2=0,0,0
    for file in train_docs:
        score_doc1[file]=doc_sim(file,did1) # similarity score b/w two documents
        score_doc2[file]=doc_sim(file,did2)
    score_doc1=dict(sorted(score_doc1.items(), key=lambda item: item[1],reverse=True)) # sort documents based on similarity score
    score_doc2=dict(sorted(score_doc2.items(), key=lambda item: item[1],reverse=True)) # sort documents based on similarity score
    list1 = list(score_doc1.keys())[:7]
    list2 = list(score_doc2.keys())[:7]
    #NN1.append(list1)
    #NN2.append(list2)
    for i in range(len(list1)):
        sim_score=[]
        for j in range(len(list2)):
            sim_score.append(sim_nn_docs(list1[i],list2[j]))
        similarity_score_1=similarity_score+max(sim_score,default=0)
    for i in range(len(list2)):
        sim_score=[]
        for j in range(len(list1)):
            sim_score.append(sim_nn_docs(list2[i],list1[j]))
        similarity_score_2=similarity_score+max(sim_score,default=0)
    similarity_score_1=float(similarity_score_1/len(list1))
    similarity_score_2=float(similarity_score_2/len(list2))
    similarity_score=float((similarity_score_1+similarity_score_2)/2)
    return similarity_score
def sim_test_docs():
    sim_score=0
    for i in range(0,50):
        for j in range(i+1,50):
            sim_score=document_sim(test_docs[i],test_docs[j])
            print(test_docs[i],test_docs[j],"   ",sim_score)
sim_test_docs()
#print(NN1)
#print(NN2)


            

