'''

    Algorithm for lstm
    ref:https://medium.com/@gautam.karmakar/manhattan-lstm-model-for-text-similarity-2351f80d72f1

'''

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout
from keras.layers import Concatenate
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import Layer
import tensorflow as tf
from numpy import shape
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from keras import optimizers
import keras as ker
import numpy as np
class Siamese_LSTM:

    def train_model(self,X,Y,validation_size,embeddings,embedding_dim,max_seq_length):
        
        #Disable gpu
        tf.config.experimental.set_visible_devices([], 'GPU')

        # Load training set

        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
        X_train = self.split_and_zero_padding(X_train, max_seq_length)
        X_validation = self.split_and_zero_padding(X_validation, max_seq_length)

        gpus = 2
        batch_size = 1#1024 * gpus
        n_epoch = 10
        n_hidden = 50

        x = Sequential()

        x.add(Embedding(len(embeddings), embedding_dim,weights=[embeddings], input_shape=(max_seq_length,), trainable=False))

        x.add(LSTM(n_hidden,input_shape=(max_seq_length,)))

        shared_model = x
        #meghan@rooman.net
        # The visible layer
        left_input = Input(shape=(max_seq_length,), dtype='int32')
        right_input = Input(shape=(max_seq_length,), dtype='int32')

        # Pack it all up into a Manhattan Distance model
        malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
        model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

        
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        model.summary()
        shared_model.summary()

        model.compile(loss='mean_squared_error', optimizer='sgd')

        
        malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                                batch_size=batch_size, epochs=n_epoch,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))
        
        model.save('lstmtest1.model')
    
    def split_and_zero_padding(self,df, max_seq_length):
        # Split to dicts
        X = {'left': df['sentences1_n'], 'right': df['sentences2_n']}

        # Zero padding
        for dataset, side in itertools.product([X], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

        return dataset
    
        
    def compare(self,df):
        model = ker.models.load_model('lstmtest1.model',custom_objects={'ManDist': ManDist})
        max_seq_length =179
        X_test = self.split_and_zero_padding(df, max_seq_length)
        prediction = model.predict([X_test['left'], X_test['right']])
        return prediction
    
    
    def exponent_neg_manhattan_distance(left, right):
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
