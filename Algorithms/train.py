"""

Program for training different models.

"""

from Sample_Data.data import sample_data
from pre_processing import pre_processing
from word2vec import Word2Vec
from lstm import Siamese_LSTM

import numpy as np
import pandas as pd


# Getting titles and abstracts

titles=(sample_data.keys())
abstracts=sample_data.values()

# creating tokens
prep=pre_processing()
title_tokens=[prep.advanced_ops(title) for title in titles]
abstracts_tokens=[prep.advanced_ops(abstract) for abstract in abstracts]

# Training word2vec model

wtv=Word2Vec()
wtv.train_word2vec_model(abstracts_tokens)

# Training lstm model

TRAIN_CSV = 'C:\\Users\\bilwa\\code\\ProjectUniquenessChecker\\Algorithms\\sample_data.csv'
df = pd.read_csv(TRAIN_CSV)
embedding_dim = 300
train_df,embeddings=wtv.get_df_embs(df,embedding_dim)

# Split into train and test 
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['sentences1_n', 'sentences2_n']]
Y = train_df['is_similar']

max_seq_length=300

lstm_train=Siamese_LSTM()
lstm_train.train_model(X,Y,validation_size,embeddings,embedding_dim,max_seq_length)