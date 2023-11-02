import math
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
import math
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score
import os
import random
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
bert_embedding=open("/home/subinay/Documents/data/pairwise_similarity/K-fold/supervised_non_parametric/sentence_legal_bert_uncased_embedding_200_docs.json")
embedding= json.load(bert_embedding)
# list of tags
all_tag=['life_imprisonment','murder_on_parole','second_murder','physical_assault','rarest_of_the_rare_case','death_sentence','homicide_not_murder','homicide_murder','political_rivalry','riot','juvenile_case','revenge','property_dispute','evidence_inconsistency','evidence_insufficient','prosecutorial_delay_or_inability','investigation_agency','witness_testimony','expert_witness_testimony','testimony_challenged']

                      ################### Computing Ground Truth ###########################
#Ground_truth building by computing tag_similarity  and computing section similarity for a document-pair 
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

         #################### Computing Document-pair similarity using BERT embedding #################  

def doc_sim(did1,did2): 
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
    annotated_score=max(score1)
    return annotated_score

def sim_test_docs(test_docs):
    sim_score=0
    for i in range(0,len(test_docs)):
    	for j in range(i+1,len(test_docs)):
    		sim_score=doc_sim(test_docs[i],test_docs[j])
    		str=test_docs[i]+test_docs[j] #store pair id
    		if str in pair_id:  # to check pair in ground truth file
    			sim_value.append(float(sim_score))
    			doc2.append(str)
    			pair_sim_train[str]=float(sim_score)

        ############################## Evaluation ###################################

# Measuering correlation between generated documnet-pair similarity and ground-truth similarity 
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
        test_docs=files[start:start+40] # 200 documents using 5-fold cross validation
        tag_sim(test_docs) 
        jac_sim()
        sim_test_docs(test_docs)
        correlation()
        start=start+40
    print("###################################")
    print("kendall",float(sum(kendall)/5))
    print("spearman",float(sum(spearman)/5))
    print("Pearson",float(sum(pearson)/5))
    print("rbo",float(sum(rank)/5))
k_fold()