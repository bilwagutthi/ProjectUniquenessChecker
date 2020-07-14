"""
Program for initial training of models
"""
from variables import MAX_TITLE_LENGHT, DIMENSIONS
from variables import WORD_VEC_MODEL
from variables import TITLE_MODEL_JSON, TITLE_MODEL_H5
from variables import TITLE_ARCH1, TITLE_ARCH2, TILE_PLOT

import logging
import time
import pandas as pd
from numpy import random,shape, array
from flask import Flask

import gensim.downloader as api
from gensim.parsing.preprocessing import remove_stopwords , preprocess_string
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from keras.preprocessing.sequence import pad_sequences

from Algorithms.SiMantIcepLSTM import SiInceptionLSTM
from Algorithms.Sent2Vec import getSentVecs
from MetaData.Datasets.datasets import datasets
from models import db, Projects, Colleges
from config import SQLALCHEMY_DATABASE_URI

logging.basicConfig(level=logging.INFO,filename='MetaData/'+'traininginfo.txt', format='%(asctime)s :: %(levelname)s :: %(message)s')
def lognprint(message):
    logging.info(message)
    print(message)

lognprint('*'*50)
lognprint('\n\tTRAINING BEGAINS\n')
train_start_time=time.time()

lognprint('\n Creating database projects.db\n\nAdding rows to db.\n\n')
titles=list(datasets.keys())
abstracts=list(datasets.values())
app = Flask(__name__)
app.config.from_pyfile('config.py')
with app.test_request_context():
    db.init_app(app)
    db.create_all()
    user1 = Colleges('test1','test1','test1','test1@email.com','test1 college,test 1 street, test city','1122334455')
    db.session.add(user1)
    for i in range(0,33):
        if i < 11 : user1.projects.append(Projects(titles[i],abstracts[i],'2017','CSE'))
        elif i < 22 : user1.projects.append(Projects(titles[i],abstracts[i],'2018','CSE'))
        else:user1.projects.append(Projects(titles[i],abstracts[i],'2019','CSE'))
    
    user2 = Colleges('test2','test2','test2','test2@email.com','test2 college,test 2 street, test city','6677889900')
    db.session.add(user2)
    for i in range(33,66):
        if i < 44 : user2.projects.append(Projects(titles[i],abstracts[i],'2017','CSE'))
        elif i < 55 : user2.projects.append(Projects(titles[i],abstracts[i],'2018','CSE'))
        else:user2.projects.append(Projects(titles[i],abstracts[i],'2019','CSE'))
    user3 = Colleges('test3','test3','test3','test3@email.com','test3 college,test 3 street, test city','234567890')
    db.session.add(user3)
    for i in range(66,100):
        if i < 77 : user3.projects.append(Projects(titles[i],abstracts[i],'2017','CSE'))
        elif i < 88 : user3.projects.append(Projects(titles[i],abstracts[i],'2018','CSE'))
        else:user3.projects.append(Projects(titles[i],abstracts[i],'2019','CSE'))
    db.session.commit()


lognprint('\n\n\nLoading gensims corpus and adding the vocab to our word_model\n')
temptime=time.time()
word_model=Word2VecKeyedVectors(vector_size=DIMENSIONS)
corpus_model=Word2Vec(api.load('text8'),size=DIMENSIONS)
corpus_words=list(corpus_model.wv.vocab.keys())
corpus_vectors=[corpus_model.wv[word] for word in corpus_words]
word_model.add(corpus_words,corpus_vectors)
lognprint("Finished loading gensim's corpus model.\nTime taken:{t}\n".format(t=time.time()-temptime))
lognprint("Creating data-frames of Word and Sentence Trainers\n")

msrcsv='MetaData/'+'MSRTrainData.csv'
leecsv='MetaData/'+'LeeDocSimTrain.csv'
tit_df=pd.read_csv(msrcsv, error_bad_lines=False)
abs_df=pd.read_csv(leecsv, error_bad_lines=False)

lognprint('Loading words to re-train word model\n')
new_words_list=[]
for index,row in tit_df.iterrows():
    for i in [row['Sentence1'],row['Sentence2']]:
        new_words_list.append(preprocess_string( remove_stopwords(i)))
               
for index,row in abs_df.iterrows():
    for i in [row['Document1'],row['Document2']]:
        new_words_list.append(preprocess_string( remove_stopwords(i)))

for i in titles:new_words_list.append(preprocess_string( remove_stopwords(i)))
for i in abstracts:new_words_list.append(preprocess_string( remove_stopwords(i)))

lognprint('Re-training with Word2Vec model with new words\n')
temp_time=time.time()
new_model = Word2Vec(new_words_list, size=DIMENSIONS, window=5, min_count=1, workers=4)
lognprint('Finished temporary model with new words, adding words to word2vec model.\nTime taken {t}\n'.format(t=time.time()-temp_time))
word_vecs=[]
words=[]
for lis in new_words_list:
    for word in lis:
        words.append(word)
        word_vecs.append(new_model.wv[word])
word_model.add(words,word_vecs,replace=False)
word_model.save("MetaData/"+WORD_VEC_MODEL)
lognprint("Finished training Word2Vec Model and saved.\nTotal vocabulary size {vocab_size}\n".format(vocab_size=len(word_model.vocab)))

lognprint("\n\n\n Starting with training neural network models\n")
lognprint('Creating list of word2vec array for training\n')
word_model.init_sims(replace=False)
Y_title=tit_df['Score']
X1_Title=[]
X2_Title=[]
for index, row in tit_df.iterrows():
    sentence1=row['Sentence1']
    sentence2=row['Sentence2']
    tokens1=preprocess_string( remove_stopwords(sentence1) )
    tokens2=preprocess_string( remove_stopwords(sentence2) )
    if tokens1==[]:tokens1=['print']
    if tokens2==[]:tokens2=['print']
    vec_matrix1=[]
    for word in tokens1:
        vec=0
        try:vec=word_model.wv[word]
        except:vec=random.rand(DIMENSIONS)
        vec_matrix1.append(vec)
    vec_matrix2=[]
    for word in tokens2:
        vec=0
        try:vec=word_model.wv[word]
        except:vec=random.rand(DIMENSIONS)
        vec_matrix2.append(vec)
    temp1=[vec_matrix1]
    temp2=[vec_matrix2]
    pad_vec_matrix1=pad_sequences(temp1, padding='post', truncating='post', maxlen= MAX_TITLE_LENGHT,dtype='float64')
    X1_Title.append(pad_vec_matrix1[0])
    pad_vec_matrix2=pad_sequences(temp2, padding='post', truncating='post', maxlen= MAX_TITLE_LENGHT,dtype='float64')
    X2_Title.append(pad_vec_matrix2[0])

tile_model= SiInceptionLSTM()

tile_model.build_model(x1=X1_Title, x2=X2_Title, Y=Y_title,
                        max_seq_length=MAX_TITLE_LENGHT,embedding_dim=DIMENSIONS,
                        arch_file_name1=TITLE_ARCH1 , arch_file_name2=TITLE_ARCH2,
                        plot_filename= TILE_PLOT,
                        modeljsonfile= TITLE_MODEL_JSON, modelh5file=TITLE_MODEL_H5)

lognprint('Training of Title model Finished')


lognprint("\n\nTotal training time : {t}\n".format(t=time.time()-train_start_time))
lognprint("Training of Word2Vec model and title model . Models can be found in the meta file\n")
lognprint("\n\n\n\t\t\t Training Completed\n\n")
lognprint('*'*50)