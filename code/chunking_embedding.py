import torch
import json
import os
import numpy as np
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from transformers import LongformerTokenizer, LongformerModel
# Load the Longformer tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
# tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
#tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
#model = AutoModel.from_pretrained("law-ai/InLegalBERT")
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained("bert-base-uncased")
#tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
#model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
dir="/home/subinay/Documents/data/pairwise_similarity/sentence_doc/"
files=os.listdir(dir)
document_embedding={}

###### BERT embedding for every chunk of each document ###########
def chunk_to_bert_vector(text):
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    x=output[0].tolist()
    tensor=x[0][0]
    return tensor

##### Chunking every document based on token size limitation ###########
def chunking(docid):
  print(docid)
  text=open("/home/subinay/Documents/data/three_level_processed/preprocessed_file2/"+docid)
  set_chunks=[]
  chunk=''
  document=""
  for sent in text:
     sent=sent.strip()
     document=document+" "+sent
  document=document.split(" ")
  number_chunks=int(len(document)/400)
  start=0
  ## basically this portion is used to construct every chunk, where some of the words are added from previous chunk.
  for i in range(0,number_chunks+1):  
      chunk=document[start:start+450]
      chunk=" ".join(chunk)
      set_chunks.append(chunk[1:])
      start=start+400  
  return set_chunks

### Construction a collection of embedding of each document chunk-wise
def document_to_bert_vector(docid):
    set_of_chunks=chunking(docid) ## Collection of chunks of particular document
    chunk_embeddings = []
    print(set_of_chunks)
    for chunk in set_of_chunks:
        print(chunk)
    #     embeddings=chunk_to_bert_vector(chunk) ## Output of BERT embedding for every chunk
    #     chunk_embeddings.append(embeddings)
    # document_embedding[docid]=chunk_embeddings ## Stored embedding of every chunk in a dictionary

def start():
    for f in files:
        document_to_bert_vector(f)
    with open("chunk_inLegalBERT_embedding_200_docs.json", "w") as outfile:
        json.dump(document_embedding, outfile)

start()












# def document_to_bert_vector(str):
#     doc1=[]
#     with open("/home/subinay/Documents/data/pairwise_similarity/sentence_doc/"+str,"r") as f1:
#         for line in f1:
#             line=line.strip()
#             doc1.append(line)
#     sentence_embeddings = []
#     for i in range(len(doc1)):
#         vector1 = sentence_to_bert_vector(doc1[i])
#         sentence_embeddings.append(vector1)
#     #doc_embedding=[sum(sub_list) / len(sub_list) for sub_list in zip(*sentence_embeddings)] # average of sentence-embedding
#     dict[str]=sentence_embeddings
#     #dict[str]=doc_embedding
# dir="/home/subinay/Documents/data/pairwise_similarity/40_docs/"
# files=os.listdir(dir)
# for f in files:
#     document_to_bert_vector(f)
