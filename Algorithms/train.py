"""

Program for training different models.

"""

from Sample_Data.data import sample_data
from pre_processing import pre_processing
from word2vec import Word2Vec
from lstm import Siamese_LSTM

import numpy as np
import pandas as pd


# getting titles and abstracts

titles=(sample_data.keys())
abstracts=sample_data.values()

# creating tokens
prep=pre_processing()
title_tokens=[prep.advanced_ops(title) for title in titles]
abstracts_tokens=[prep.advanced_ops(abstract) for abstract in abstracts]



# Training word2vec model

wtv=Word2Vec()
#wtv.train_word2vec_model(abstracts_tokens)

# Training lstm model

# Pre- paring training data
TRAIN_CSV = 'C:\\Users\\bilwa\\code\\ProjectUniquenessChecker\\Algorithms\\sample_data.csv'
df = pd.read_csv(TRAIN_CSV)
embedding_dim = 300
train_df,embeddings=wtv.get_df_embs(df,embedding_dim)

print(train_df.columns,train_df[0:6])
        # Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['sentences1_n', 'sentences2_n']]
Y = train_df['is_similar']

print(X[0:2],Y[0:2])

# Getting word embeddings


use_w2v = True

#embeddings =wtv.trained_vecs(abstracts_tokens,embedding_dim)
print('Before:',type(embeddings),len(embeddings))
# LSTM training
max_seq_length =max([len(token_list) for token_list in abstracts_tokens])

print("max lenght",max_seq_length)

lstm_train=Siamese_LSTM()
lstm_train.train_model(X,Y,validation_size,embeddings,embedding_dim,max_seq_length)