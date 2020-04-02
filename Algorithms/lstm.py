'''

    Algorithm for lstm
    ref:https://medium.com/@gautam.karmakar/manhattan-lstm-model-for-text-similarity-2351f80d72f1

'''

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout, Concatenate, Layer
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import keras as ker

import tensorflow as tf
from numpy import shape
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Siamese_LSTM:

    def train_model(self,X,Y,validation_size,embeddings,embedding_dim,max_seq_length,batch_size,n_epoch):
        
        #Disable gpu
        tf.config.experimental.set_visible_devices([], 'GPU')

        # Load training set

        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
        X_train = self.split_and_zero_padding(X_train, max_seq_length)
        X_validation = self.split_and_zero_padding(X_validation, max_seq_length)

        n_hidden = 50

        x = Sequential()

        x.add(Embedding(len(embeddings), embedding_dim,weights=[embeddings], input_shape=(max_seq_length,), trainable=False))

        x.add(LSTM(n_hidden,input_shape=(max_seq_length,)))

        shared_model = x
        # The visible layer
        left_input = Input(shape=(max_seq_length,), dtype='int32')
        right_input = Input(shape=(max_seq_length,), dtype='int32')

        # Pack it all up into a Manhattan Distance model
        malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
        model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
        
        adam_optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        
        model.compile(loss='mean_squared_error', optimizer=adam_optimizer, metrics=['accuracy'])
        model.summary()
        shared_model.summary()

        
        malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                                batch_size=batch_size, epochs=n_epoch,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))
        
        model.save('lstmtest1.model')

        # Plotting and saving data
        # Plot accuracy
        plt.subplot(211)
        print(malstm_trained.history.keys())
        plt.plot(malstm_trained.history['accuracy'])
        plt.plot(malstm_trained.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot loss
        plt.subplot(212)
        plt.plot(malstm_trained.history['loss'])
        plt.plot(malstm_trained.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout(h_pad=1.0)
        plt.savefig('history-graph.png')

        print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
            "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
        print("Done.")
    
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
