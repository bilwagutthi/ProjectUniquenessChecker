'''
Program to generate word2vec vectors

'''

import gensim
import numpy as np
from Algorithms.pre_processing import pre_processing
from numpy import random
from keras.preprocessing.sequence import pad_sequences

def list_wordvecs(model,var_list,max_seq_length,dimensions):
    pp=pre_processing()
    new=[]
    for v in var_list:
        v=pp.advanced_ops(v)
        vec_matrix1=[]
        for word in v:
            vec=0
            try:vec=model.wv[word]
            except:vec=np.zeros(dimensions)#random.rand(dimensions)
            vec_matrix1.append(vec)
        temp1=[vec_matrix1]
        pad_vec_matrix1=pad_sequences(temp1, padding='post', truncating='post', maxlen=max_seq_length ,dtype='float64')
        new.append(pad_vec_matrix1[0])
    return new

def get_wordvecs(model,tokens,max_seq_lenght,dimensions):
    vec_matrix1=[]
    if tokens==[]:tokens=['p']
    for word in tokens:
        vec=0
        try:vec=vec=model.wv[word]
        except:vec=np.zeros(dimensions)#random.rand(dimensions)
        vec_matrix1.append(vec)
    temp1=[vec_matrix1]
    pad_vec_matrix1=pad_sequences(temp1, padding='post', truncating='post', maxlen=max_seq_lenght ,dtype='float64')
    return pad_vec_matrix1


