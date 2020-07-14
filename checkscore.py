from flask import Flask, render_template, request, redirect , url_for, jsonify

import keras.backend as K
from keras.layers import Layer
from keras.optimizers import Adadelta
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.backend import set_session
import keras.backend as K
import keras.backend.tensorflow_backend as tb

from gensim.parsing.preprocessing import remove_stopwords , preprocess_string
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors

import spacy

from variables import WORD_VEC_MODEL,TITLE_MODEL_H5, TITLE_MODEL_JSON
from variables import MAX_ABSTRACT_LENGHT, MAX_TITLE_LENGHT, DIMENSIONS
from Algorithms.word2vec import list_wordvecs , get_wordvecs
from Algorithms.pre_processing import pre_processing
from Algorithms.Sample_Data.data import sample_data 
from models import db ,Projects, Colleges

from time import perf_counter
import tensorflow as tf
#from Sample_Data.test_data import title,abstract
from numpy import shape , array , matrix
import numpy as np
#import tensorflow.backend as backend


print('\n\n Loading Server \n\n')
app=Flask(__name__)
app.config.from_pyfile('config.py')
db.init_app(app)

class ManDist(Layer):
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

optimizer= Adadelta()

class TitleModel:
    def __init__(self):
        json_file = open("MetaData/"+ TITLE_MODEL_JSON , 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        title_model = model_from_json(loaded_model_json,custom_objects ={'ManDist': ManDist})
        title_model.load_weights('MetaData/'+TITLE_MODEL_H5)
        title_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error'])
        self.model = title_model
        self.model._make_predict_function()
        self.session = tb.get_session()
        self.graph = tf.get_default_graph()
    def predict(self,values):
        with self.session.as_default():
            with self.graph.as_default():
                pred = self.model.predict(values)
        return(pred)

title_model=TitleModel()

# loading outer data
def previous():
    with app.app_context():
        temp=Projects.query.with_entities(Projects.title).all()
        titles=[i[0] for i in temp]
        temp=Projects.query.with_entities(Projects.abstract).all()
        abstracts=[i[0] for i in temp]
        return titles,abstracts

previous_titles, previous_abstracts= previous()

#Load WordVec models
word_model = Word2VecKeyedVectors.load("MetaData/"+WORD_VEC_MODEL)
word_model.init_sims(replace=False)
pp=pre_processing()
#Get word vectors of previous titles
title_vecList=list_wordvecs(model=word_model, var_list=previous_titles,max_seq_length=MAX_TITLE_LENGHT, dimensions=DIMENSIONS)

nlp = spacy.blank('en')
keys = []
for idx in range(len(word_model.vocab)):
    keys.append(word_model.index2word[idx])
nlp.vocab.vectors = spacy.vocab.Vectors(data=word_model.syn0, keys=keys)

def get_score(test_title,test_abstract,title_compare,wordmodel,titlemodel,dataset_titles,dataset_abstracts):
    test_title_tokens = pp.advanced_ops(test_title)
    test_title_vectors= get_wordvecs(model=wordmodel,tokens=test_title_tokens,max_seq_lenght=MAX_TITLE_LENGHT,dimensions=DIMENSIONS)
    titles_dist_0=[]
    for title in title_compare:
        pred=titlemodel.predict([array([title]),array(test_title_vectors)])
        print(pred,end=' ')
        titles_dist_0.append(pred[0][0])
    norm=1.5182762
    titles_dist=[]  
    for dist in titles_dist_0:
        x=dist/norm
        titles_dist.append(round(x*100,5))
        print(x,end=' ')
    
    top3titles=sorted(zip(titles_dist,dataset_titles),reverse=True)[:3]
    
    abs_dist=[]
    for abstract in dataset_abstracts:
        doc1=nlp(remove_stopwords(abstract))
        doc2=nlp(remove_stopwords(test_abstract))
        abs_dist.append(doc1.similarity(doc2)*100)
        print(doc1.similarity(doc2),end='')
    top3abstracts=sorted(zip(abs_dist,dataset_titles),reverse=True)[:3]
    max_sim=max(max(top3abstracts),max(top3titles))
    
    uniqueness=100- max_sim[0] 
    top_title=max_sim[1]

    return (uniqueness,top_title,top3titles,top3abstracts)
    


@app.route("/", methods=['GET', 'POST'])
@app.route("/score", methods=['GET', 'POST'])
def score():
    if request.method == 'GET':
        return render_template('score.html')

    if request.method == 'POST':
        title = request.form.get('title') 
        abstract = request.form.get('abstract')
        result= get_score(test_title=title, test_abstract=abstract, 
                           title_compare= title_vecList, wordmodel=word_model,
                           titlemodel=title_model,
                           dataset_titles=previous_titles,dataset_abstracts=previous_abstracts)
        
        sim_titles={i:{'score':result[2][i][0],'title':result[2][i][1]} for i in range(3)}
        sim_abstracts={i:{'score':result[3][i][0],'title':result[3][i][1]} for i in range(3)}
        response={
            'score':result[0],
            'similarmax':result[1],
            'sim_titles':sim_titles,
            'sim_abstracts':sim_abstracts
        }

        return jsonify(response)

if __name__=="__main__":
    app.run(debug=True,threaded=False)