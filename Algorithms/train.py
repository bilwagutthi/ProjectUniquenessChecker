"""

Program for training different models.

"""

from Sample_Data.data import sample_data
from pre_processing import pre_processing
from word2vec import Word2Vec
from lstm import Siamese_LSTM

import numpy as np


# getting titles and abstracts

titles=(sample_data.keys())
abstracts=sample_data.values()

# creating tokens
prep=pre_processing()
title_tokens=[prep.advanced_ops(title) for title in titles]
abstract_tokens=[prep.advanced_ops(abstract) for abstract in abstracts]



# Training word2vec model

wtv=Word2Vec()
#wtv.train_word2vec_model(abstract_tokens)

# Training lstm model

# Getting word embeddings
embedding_dim = 300
max_seq_length = 20
use_w2v = True

embeddings =np.array([wtv.get_vecs(abstract_token, embedding_dim=embedding_dim) for abstract_token in abstract_tokens])

# LSTM training

lstm_train=Siamese_LSTM()
lstm_train.train_model(embeddings,embedding_dim,max_seq_length)
