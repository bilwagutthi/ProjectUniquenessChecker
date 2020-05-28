"""
Program for initial training of models
"""

from variables import WORD_VEC_MODEL
from variables import TITLE_MODEL_JSON, TITLE_MODEL_H5
from variables import ABSTRACT_MODEL_H5, ABSTRACT_MODEL_JSON
from variables import MAX_ABSTRACT_LENGHT, MAX_TITLE_LENGHT, DIMENSIONS
from variables import TITLE_ARCH1, TITLE_ARCH2, TILE_PLOT
from variables import ABSTRACT_ARCH1, ABSTRACT_ARCH2, ABSTRACT_PLOT

import logging
import time
import pandas as pd
from numpy import random,shape, array

import gensim.downloader as api
from gensim.parsing.preprocessing import remove_stopwords , preprocess_string
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from keras.preprocessing.sequence import pad_sequences

from Algorithms.SiMantIcepLSTM import SiInceptionLSTM
from Algorithms.Sent2Vec import getSentVecs

logging.basicConfig(level=logging.INFO,filename='MetaData/'+'traininginfo.txt', format='%(asctime)s :: %(levelname)s :: %(message)s')
def lognprint(message):
    logging.info(message)
    print(message)


lognprint('*'*50)
lognprint('\n\tTRAINING BEGAINS\n')
train_start_time=time.time()
lognprint('Loading gensims corpus and adding the vocab to our word_model')
temptime=time.time()
word_model=Word2VecKeyedVectors(vector_size=DIMENSIONS)
corpus_model=Word2Vec(api.load('text8'),size=DIMENSIONS)
corpus_words=list(corpus_model.wv.vocab.keys())
corpus_vectors=[corpus_model.wv[word] for word in corpus_words]
word_model.add(corpus_words,corpus_vectors)
lognprint("Finished loading gensims corpus model.\nTime taken:{t}".format(t=time.time()-temptime))
lognprint("Creating dataframes of Word and Sentence Trainers")

msrcsv='MetaData/'+'MSRTrainData.csv'
leecsv='MetaData/'+'LeeDocSimTrain.csv'
tit_df=pd.read_csv(msrcsv, error_bad_lines=False)
abs_df=pd.read_csv(leecsv, error_bad_lines=False)

lognprint('Loading words to re-train word model')
new_words_list=[]
for index,row in tit_df.iterrows():
    for i in [row['Sentence1'],row['Sentence2']]:
        new_words_list.append(preprocess_string( remove_stopwords(i)))
               
for index,row in abs_df.iterrows():
    for i in [row['Document1'],row['Document2']]:
        new_words_list.append(preprocess_string( remove_stopwords(i)))

lognprint('Re-training with Word2Vec model with new words')
temp_time=time.time()
new_model = Word2Vec(new_words_list, size=DIMENSIONS, window=5, min_count=1, workers=4)
lognprint('Finished temporary model with new words, adding words to word2vec model.\nTime taken {t}'.format(t=time.time()-temp_time))
word_vecs=[]
words=[]
for lis in new_words_list:
    for word in lis:
        words.append(word)
        word_vecs.append(new_model.wv[word])
word_model.add(words,word_vecs,replace=False)
word_model.save("MetaData/"+WORD_VEC_MODEL)
lognprint("Finished training Word2Vec Model and saved.\nTotal vocabulary size {vocab_size}".format(vocab_size=len(word_model.vocab)))

lognprint("\n\n\n Starting with training neural network models")
lognprint('\nCreating list of word2vec array for training')
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

X1_abs=[]
X2_abs=[]
Y_abs=abs_df['Similarity']
lognprint('\n\nGetting abstract embeddings')
for index,row in abs_df.iterrows():
    X1_abs.append(getSentVecs(paragraph=row['Document1'],word_model=word_model,dimensions=DIMENSIONS,max_seq_len=MAX_ABSTRACT_LENGHT))
    X2_abs.append(getSentVecs(paragraph=row['Document2'],word_model=word_model,dimensions=DIMENSIONS,max_seq_len=MAX_ABSTRACT_LENGHT))

abstract_model=SiInceptionLSTM()
abstract_model.build_model(x1=X1_abs, x2=X1_abs, Y=Y_abs,
                        max_seq_length=MAX_ABSTRACT_LENGHT,embedding_dim=DIMENSIONS,
                        arch_file_name1=ABSTRACT_ARCH1 , arch_file_name2=ABSTRACT_ARCH2,
                        plot_filename= ABSTRACT_PLOT,
                        modeljsonfile= ABSTRACT_MODEL_JSON, modelh5file=ABSTRACT_MODEL_H5)

lognprint("Training of Abstract Model Finished")
lognprint("\n\nTotal training time : {t}".format(t=time.time()-train_start_time))
lognprint("Training of Word2Vec model, title model and abstract model finished. Models can be found in the meta file")
lognprint("\t\t\t Training Completed\n\n")
lognprint('*'*50)