import random
import os
import json
import re
import numpy as np
from numpy.linalg import norm
import math
from collections import Counter
from sklearn.metrics import jaccard_score
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import numpy as np
from numpy.linalg import norm
import math
import json
from scipy import stats
from scipy.stats import spearmanr
import math
import rbo
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score
from bs4 import BeautifulSoup
pair_id=[]
gt_value=[]
sim_value=[]
pair_sim={}
pair_sim_train={}
doc1=[]
doc2=[]

# shuffle the list of 200 documents
SEED = 42
dir="/home/subinay/Documents/data/Tag_prediction_model/clean_random_200_doc/"
files=os.listdir(dir)
random.seed(SEED)
random.shuffle(files)
stop_words=[]
## Annotated span along with tags
json_data=open("/home/subinay/Documents/data/sentence_tag/search_file_new_2.json") 
jdata = json.load(json_data)
## tf_idf values of each document
# json_embedding_tf_idf=open("/home/subinay/Documents/data/pairwise_similarity/K-fold/tf_idf_10000_200/tf_idf_embedding_200_docs_1.json")
# embedding= json.load(json_embedding_tf_idf)
bert_embedding=open("/home/subinay/Documents/data/pairwise_similarity/K-fold/supervised_non_parametric/sentence_InLegalbert_uncased_embedding_200_docs.json")
embedding= json.load(bert_embedding)
## Sections information of each document
json_embedding_section=open("/home/subinay/Documents/data/pairwise_similarity/supervised_non_prarmetric/section_embedding_200docs.json")
embedding_section= json.load(json_embedding_section)
WORD = re.compile(r"\w+")

# list of tags
all_tag=['life_imprisonment','murder_on_parole','second_murder','physical_assault','rarest_of_the_rare_case','death_sentence','homicide_not_murder','homicide_murder','political_rivalry','riot','juvenile_case','revenge','property_dispute','evidence_inconsistency','evidence_insufficient','prosecutorial_delay_or_inability','investigation_agency','witness_testimony','expert_witness_testimony','testimony_challenged']

      #################  Assesing Ground-Truth  based on tag informartion for each doument-pair ####################
file1_a=[]
list_section=[]
list_tag=[]
sec_sim=[]
def jac_sim():
    k=len(file1_a)
    for i in range(0,k):    #(k,86)
        for j in range(i+1,k):
            if(i!=j):
                #sec_score=jaccard_score(list_section[i],list_section[j])
                #sec_sim.append(sec_score)
                tag_score=jaccard_score(list_tag[i],list_tag[j])
                str=file1_a[i]+file1_a[j]
                if(float(tag_score)!=0):
                    pair_id.append(str)
                    doc1.append(str)
                    pair_sim[str]=float(tag_score)
                    gt_value.append(float(tag_score))
                #print(z)
                #print(file1_a[i],file1_a[j]," ",z)

def tag_sim(test_docs):
    for f in test_docs:
        with open("/home/subinay/Documents/data/pairwise_similarity/K-fold/200_docs_section/"+f,"r") as f1:
            for line in f1:
                if "Indian Penal Code, 1860_302" in line:
                    list2=[]
                    for i in range(0,512):
                        list2.insert(i,0)
                    file1_a.append(f)
                    tokens=line.split("#") #
                    tokens=tokens[:-1]   #remove the last element
                    for token in tokens:
                        token= token.replace("Indian Penal Code, 1860_","") 
                        list2[int(token)]=1
        with open("/home/subinay/Documents/data/sentence_tag/Annotated_Anjana_200/"+f,"r") as f2:
            content = f2.read();
        result = " ".join(line.strip() for line in content.splitlines())
        soup = BeautifulSoup(result,'html.parser')
        tags = [tag.name for tag in soup.find_all()]
        tags1=set(tags)
        with open("/home/subinay/Documents/data/sentence_tag/Annotated_Bipasha_200/"+f,"r") as f2:
            content = f2.read();
        result = " ".join(line.strip() for line in content.splitlines())
        soup = BeautifulSoup(result,'html.parser')
        tags = [tag.name for tag in soup.find_all()]
        tags2=set(tags)
        myset = tags1 | tags2
        tag=list(myset)
        list3=[]
        for j in range(0,20):
            list3.insert(j,0)
        for i in range(len(tag)):
            index=all_tag.index(tag[i])
            list3[index]=1
        for i in list3:
            list2.append(i)
        list_tag.append(list2[512:]) # created list of tags 
        #list_section.append(list2[0:512]) # created list of sections



           ############  Document Similarity using supervised-non-parametric approach  ##############

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

def doc_sim(did1,did2): # used for BERT embedding
    sentence_embeddings1=embedding[did1]
    sentence_embeddings2=embedding[did2]
    annotated_score,score2=0,0
    score1=[]
    for i in range(len(sentence_embeddings1)):
        sentence_score=[]
        A=np.array(sentence_embeddings1[i])
        for j in range(len(sentence_embeddings2)):
            B=np.array(sentence_embeddings2[j])
            score = np.dot(A,B)/(norm(A)*norm(B))
            sentence_score.append(score)
        max_score=max(sentence_score)
        score1.append(max_score)
    bert_sim_score=max(score1)
    return bert_sim_score

# def doc_sim(docid1,docid2): # compute similarity b/w two documents using cosine similarity (used for tf-idf values)
#     score,score_tf_idf,score_section=0,0,0
#     doc1_embedding=embedding[docid1]
#     doc2_embedding=embedding[docid2]
#     # A=np.array(doc1_embedding)
#     # B=np.array(doc2_embedding)
#     # score_tf_idf= np.dot(A,B)/(norm(A)*norm(B))
#     score_tf_idf=compute_cosine(doc1_embedding,doc2_embedding)
#     doc1_embedding=embedding_section[docid1]
#     doc2_embedding=embedding_section[docid2]
#     score_section=jaccard_score(doc1_embedding,doc2_embedding)
#     #print(score_tf_idf,"####",score_section)
#     score=1.0*float(score_tf_idf)+0.0*float(score_section)
#     return score 

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

def document_sim(did1,did2,train_docs):
    score_doc1={}
    score_doc2={}
    similarity_score,similarity_score_1,similarity_score_2=0,0,0
    for file in train_docs:
        score_doc1[file]=doc_sim(file,did1) # similarity score b/w two documents
        score_doc2[file]=doc_sim(file,did2)
    score_doc1=dict(sorted(score_doc1.items(), key=lambda item: item[1],reverse=True)) # sort documents based on similarity score
    #print(score_doc1)
    score_doc2=dict(sorted(score_doc2.items(), key=lambda item: item[1],reverse=True)) # sort documents based on similarity score
    list1 = list(score_doc1.keys())[:3]  # list of neighbour documents
    list2 = list(score_doc2.keys())[:3]  # list of neighbour documents
    # print(list1)
    #print(list2)
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
def sim_test_docs(test_docs,train_docs):
    sim_score=0
    for i in range(len(train_docs)):
        #print(test_docs[i])
        for j in range(i+1,len(train_docs)):
            #print(test_docs[j])
            sim_score=document_sim(test_docs[i],test_docs[j],train_docs)
            str=test_docs[i]+test_docs[j] #store pair id
            if str in pair_id: # to check pair in ground truth file
                sim_value.append(float(sim_score))
                doc2.append(str)
                pair_sim_train[str]=float(sim_score)
            #print(test_docs[i],test_docs[j],"   ",sim_score)

               ######################## Evaluation #############################

kendall=[]
pearson=[]
spearman=[]
rank=[]
def correlation():
    score_doc1=dict(sorted(pair_sim.items(), key=lambda item: item[1],reverse=True))
    gt_doc= list(score_doc1.keys())
    score_doc2=dict(sorted(pair_sim_train.items(), key=lambda item: item[1],reverse=True))
    train_doc= list(score_doc2.keys())
    list1=[]
    list2=[]
    print(len(doc1))
    print(len(doc2))
    for i in range(len(doc1)):
        index1=gt_doc.index(doc1[i])
        list1.append(index1)
        index2=train_doc.index(doc2[i])
        list2.append(index2)
    res, p_value = stats.kendalltau(list1, list2)
    res2, p_value=stats.pearsonr(gt_value,sim_value)
    corr, p_value = spearmanr(list1, list2)
    kendall.append(res)
    pearson.append(res2)
    spearman.append(corr)
    corr2=(rbo.RankingSimilarity(list1, list2).rbo())
    rank.append(corr2)
    print("Kendall",res)
    print("Spearman's correlation coefficient:", corr)
    print("Pearson",res2)
    print("Rank_biased_overlap",rbo.RankingSimilarity(list1, list2).rbo())
    file1_a.clear()
    doc1.clear()
    doc2.clear()
    gt_value.clear()
    sim_value.clear()
    sec_sim.clear()
    list_tag.clear()
    list_section.clear()
    pair_id.clear()
                 
                 ############################ Fold-wise Result ###############################
def k_fold():
    start=0
    for i in range(0,5):
        test_docs=files[start:start+40]
        givenIndex = start
        train_docs= files[:givenIndex] + files[givenIndex+40:]
        tag_sim(test_docs) 
        jac_sim()
        sim_test_docs(test_docs,train_docs)
        correlation()
        start=start+40
    print("###################################")
    print("kendall",float(sum(kendall)/5))
    print("spearman",float(sum(spearman)/5))
    print("Pearson",float(sum(pearson)/5))
    print("rbo",float(sum(rank)/5))
k_fold()

