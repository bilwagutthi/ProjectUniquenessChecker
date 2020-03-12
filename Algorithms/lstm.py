'''

    Algorithm for lstm
    ref:https://medium.com/@gautam.karmakar/manhattan-lstm-model-for-text-similarity-2351f80d72f1

'''
.
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

class Siamese_LSTM:

    def model_build():
        left_input = Input(shape=(max_seq_length,), dtype='int32')
        right_input = Input(shape=(max_seq_length,), dtype='int32')
        embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)
        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        # Since this is a siamese network, both sides share the same LSTM
        shared_lstm = LSTM(n_hidden)
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)
        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        # Pack it all up into a model
        malstm = Model([left_input, right_input], [malstm_distance])
        We need set an optimizer, I am using adadelta but any other popular optimizer such as RMSProp, Adam and even SGD could be tested to see if it increases accuracy, decreases training time by finding better local minima (yes, global minima is an elusive goal still).
        # Adadelta optimizer, with gradient clipping by norm
        optimizer = Adadelta(clipnorm=gradient_clipping_norm)
        Now we will compile and train the model.
        malstm.compile(loss=’mean_squared_error’, optimizer=optimizer, metrics=[‘accuracy’])
        # Start training
        training_start_time = time()
        malstm_trained = malstm.fit([X_train[‘left’], X_train[‘right’]], Y_train, batch_size=batch_size, nb_epoch=n_epoch, validation_data=([X_validation[‘left’], X_validation[‘right’]], Y_validation))
        
    def compare(original_abs, new_abs):

    def exponent_neg_manhattan_distance(left, right):
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
    

        