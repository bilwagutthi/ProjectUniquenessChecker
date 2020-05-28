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


from main import main , get_score
from variables import WORD_VEC_MODEL,ABSTRACT_MODEL_H5,ABSTRACT_MODEL_JSON,TITLE_MODEL_H5, TITLE_MODEL_JSON
from variables import MAX_ABSTRACT_LENGHT, MAX_TITLE_LENGHT, DIMENSIONS
from Algorithms.word2vec import list_wordvecs , get_wordvecs
from Algorithms.Sent2Vec import list_sentvecs , getSentVecs
from Algorithms.pre_processing import pre_processing
from Algorithms.Sample_Data.data import sample_data 

from time import perf_counter
import tensorflow as tf
#from Sample_Data.test_data import title,abstract
from numpy import shape , array , matrix
import numpy as np
#import tensorflow.backend as backend

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
class AbstractModel:
    def __init__(self):
        json_file = open("MetaData/"+ ABSTRACT_MODEL_JSON , 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        abstract_model = model_from_json(loaded_model_json,custom_objects ={'ManDist': ManDist})
        abstract_model.load_weights('MetaData/'+ABSTRACT_MODEL_H5)
        abstract_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error'])
        self.model = abstract_model
        self.model._make_predict_function()
        self.session = tb.get_session()
        self.graph = tf.get_default_graph()
    def predict(self,values):
        with self.session.as_default():
            with self.graph.as_default():
                pred = self.model.predict(values)
        return(pred)

title_model=TitleModel()
abstract_model=AbstractModel()
# loading outer data
previous_titles = sample_data.keys()
previous_abstracts = sample_data.values()
#Load WordVec models
word_model = Word2VecKeyedVectors.load("MetaData/"+WORD_VEC_MODEL)
word_model.init_sims(replace=False)
#Get word vectors of previous titles and abstracts
title_vecList=list_wordvecs(model=word_model, var_list=list(sample_data.keys()),max_seq_length=MAX_TITLE_LENGHT, dimensions=DIMENSIONS)
#abstract_vecList=list_wordvecs(model=word_model, var_list=list(sample_data.values()),max_seq_lenght=varb.MAX_ABSTRACT_LENGHT , dimensions=varb.DIMENSIONS)
abstract_vecList=list_sentvecs(word_model=word_model, para_list=list(sample_data.values()),max_seq_length=MAX_ABSTRACT_LENGHT , dimensions=DIMENSIONS)

def gget_score(test_title,test_abstract,title_compare,abstract_compare,wordmodel,titlemodel,abstractmodel,dataset_titles,dataset_abstracts):
    
    pp=pre_processing()
    test_title_tokens = pp.advanced_ops(test_title)
    #test_abstract_tokens = pp.advanced_ops(test_abstract)
    tim1=perf_counter()
    test_title_vectors= get_wordvecs(model=wordmodel,tokens=test_title_tokens,max_seq_lenght=MAX_TITLE_LENGHT,dimensions=DIMENSIONS)
    test_abstract_vectors=getSentVecs(paragraph=test_abstract,word_model=wordmodel,dimensions=DIMENSIONS,max_seq_len=MAX_ABSTRACT_LENGHT)
    #get_wordvecs(model=wordmodel,tokens=test_abstract_tokens ,max_seq_lenght=MAX_ABSTRACT_LENGHT,dimensions=DIMENSIONS)
    print("Time to get vectors 1 :",(perf_counter()-tim1))
    print('shape',np.shape(test_title_vectors))
    print('comp shape',np.shape(title_compare))
    print(test_title_vectors[0][0][0])
    print(title_compare[0][0][0])
    titles_dist=[]
    for title in title_compare:
        pred=titlemodel.predict([array([title]),array(test_title_vectors)])
        titles_dist.append(pred[0][0])
    top3titles=sorted(zip(titles_dist,dataset_titles),reverse=True)[:3]
    abs_dist=[]
    for absvec in abstract_compare:
        pred=abstractmodel.predict([array([absvec]),array([test_abstract_vectors])])
        abs_dist.append(pred[0][0])
    
    top3abstracts=sorted(zip(abs_dist,dataset_titles),reverse=True)[:3]

    max_sim=max(max(top3abstracts),max(top3titles))
    uniqueness=100- max_sim[0] 
    top_title=max_sim[1]

    return uniqueness,top_title,top3titles,top3abstracts
    



app=Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
@app.route("/score", methods=['GET', 'POST'])
def score():
    if request.method == 'GET':
        #return the form
        return render_template('score.html')
    if request.method == 'POST':
        #return the answer
        title = request.form.get('title') 
        abstract = request.form.get('abstract')
        print('\n'*5,"THe title and abstract is ",title,abstract,'\n'*5)
        #result= main(title, abstract)
        newpin= gget_score(test_title=title, test_abstract=abstract, 
                           title_compare= title_vecList, abstract_compare=abstract_vecList, wordmodel=word_model,
                           titlemodel=title_model, abstractmodel=abstract_model,
                           dataset_titles=previous_titles,dataset_abstracts=previous_abstracts)
        print(newpin)
        result=(86.4638939499855, 'Research is done on keyword extraction based on Word2Vec weighted TextRank', 
                [(27.0, 'Wearable smart health monitoring system for animals'),
                 (0.05, 'The Collaborative Virtual Reality Neurorobotics Lab'),
                 (0.90, 'Similarity Analysis of Law Documents Based on Word2vec')],
                [(13.536106050014496, 'Research is done on keyword extraction based on Word2Vec weighted TextRank'),
                 (0.08547345059923828, 'Efficient Video Classification Using Fewer Frames'),
                 (0.05178825813345611, 'Mobile Search Behaviors: An In-depth Analysis Based on Contexts, APPs, and Devices')
                 ])
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