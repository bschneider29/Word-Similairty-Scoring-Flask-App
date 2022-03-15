from __future__ import print_function
from flask import Blueprint, render_template,request, Response
from numpy import dot
from numpy.linalg import norm
import numpy as np
import tensorflow_hub as hub
import regex as re
from absl import logging
import time
import sys
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import math 


views = Blueprint('views', __name__)

@views.route('/',methods=["GET","POST"])

def main_page():
    if request.method=="POST":
        text_1 = request.form.get("text_1")
        text_2 = request.form.get("text_2")
        
        ##USE Model
        print('USE Processing.....', file=sys.stderr)
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        print('Model Loaded', file=sys.stderr)
        model_use = hub.load(module_url)
        print('USE Processing Complete', file=sys.stderr)
        
        def cosine(u, v):
            return dot(u, v) / (norm(u) * norm(v))

        logging.set_verbosity(logging.ERROR)  

        text_1=re.sub(r'[^\w]', ' ', text_1)
        text_2=re.sub(r'[^\w]', ' ', text_2)

        print('Clean Text 1: ', text_1, file=sys.stderr)
        print('Clean Text 2: ', text_2, file=sys.stderr)

        embeded_text_1=[model_use([text_1])[0]]
        embeded_text_2=[model_use([text_2])[0]]
        
        embeded_text_1=np.squeeze(np.asarray(embeded_text_1))
        embeded_text_2=np.squeeze(np.asarray(embeded_text_2))

        sim_use = str(round(cosine(embeded_text_1,embeded_text_2) * 100)) +'%'
        print('USE Modeling Complete', file=sys.stderr)

        print('BERT Processing.....', file=sys.stderr)
        
        bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

        sentences=[text_1,text_2]
        sentence_embeddings = bert_model.encode(sentences)
        sim_bert = str((round(cosine(sentence_embeddings[0],sentence_embeddings[1]) *100))) +'%'
        print('BERT Modeling Complete', file=sys.stderr)
        print('TF-IDF Processing.....', file=sys.stderr)

        tfidf = TfidfVectorizer()
        text_vec = tfidf.fit_transform([text_1, text_2])
        matrix = ((text_vec * text_vec.T).A)
        tfidf_sim = str(round(matrix[0,1]*100)) +'%'

        print('TF-IDF Modeling Complete', file=sys.stderr)
    else:
        sim_use='' 
        sim_bert=''
        tfidf_sim='' 
    return render_template('main.html',Sim_Score_USE='The Similarity Score for USE-Model is:  {}'.format(sim_use),Sim_Score_BERT='The Similarity Score for BERT-Model is:  {}'.format(sim_bert), Sim_Score_WORD='The Similarity Score for TF-IDF-Model is:  {}'.format(tfidf_sim))
    