import keras.backend as K
from keras.layers import Conv1D, MaxPooling1D, Input, Embedding, LSTM, Concatenate, Dense, Layer
from keras.models import Model , load_model
from keras.optimizers import Adadelta
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import matplotlib.pyplot as plt
from numpy import array

class SiInceptionLSTM():

    def build_model(self,x1,x2,Y,max_seq_length,embedding_dim,arch_file_name1,arch_file_name2,plot_filename,modeljsonfile,modelh5file):
        batchsize=64
        epochs=200
        
        left_input = Input(shape = (max_seq_length, embedding_dim))
        right_input = Input(shape = (max_seq_length, embedding_dim))
        # Building the Inception + LSTM model
        input_1 = Input(shape = (max_seq_length, embedding_dim))

        tower_1 = Conv1D(48, (1), padding='same', activation='relu')(input_1)

        tower_2=  Conv1D(48, (1), padding='same', activation='relu')(input_1)
        tower_2=  Conv1D(56, (3), padding='same', activation='relu')(tower_2)

        tower_3=  Conv1D(48, (1), padding='same', activation='relu')(input_1)
        tower_3=  Conv1D(64, (3), padding='same', activation='relu')(tower_3)
        tower_3=  Conv1D(64, (3), padding='same', activation='relu')(tower_3)

        tower_4=  MaxPooling1D((2), strides=(1), padding='same')(input_1)
        tower_4=  Conv1D(32, (1), padding='same', activation='relu')(tower_4)

        concat= Concatenate(axis = 2)([tower_1,tower_2,tower_3,tower_4])
        lstm=LSTM((embedding_dim),activation='tanh')(concat)
        incepLSTM_Model=Model(inputs=input_1,outputs=lstm)
        incepLSTM_Model.summary()
        shared_model = incepLSTM_Model

        # Building the Siamease model
        malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
        denser=Dense(1,activation='linear')(malstm_distance)
        model = Model(inputs=[left_input, right_input], outputs=[denser])

        model.summary()
        optimizer= Adadelta()
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error'])

        acrhfilepath1='MetaData/'+arch_file_name1
        acrhfilepath2='MetaData/'+arch_file_name2
        plot_model(model, to_file=acrhfilepath1 ,show_shapes=True)
        plot_model(incepLSTM_Model, to_file=acrhfilepath2,show_shapes=True)

        es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=1, patience=5,min_delta=0.005)
        HalfIncep= model.fit([x1,x2], Y,batch_size=batchsize, epochs=epochs,validation_split=0.1)
        model_json = model.to_json()
        with open("MetaData/"+ modeljsonfile , "w") as json_file:
            json_file.write(model_json)
        model.save_weights('MetaData/'+ modelh5file)
        plt.subplot(211)
        plt.plot(HalfIncep.history['mean_squared_error'])
        plt.plot(HalfIncep.history['val_mean_squared_error'])
        plt.title('Model Accuracy - Mean Squared Error ')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.subplot(212)
        plt.plot(HalfIncep.history['mean_absolute_percentage_error'])
        plt.plot(HalfIncep.history['val_mean_absolute_percentage_error'])
        plt.title('Model Accuracy - Mean Absolute Percentage Error')
        plt.ylabel('Mean Absolute Percentage Error')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.tight_layout()
        plt.savefig('MetaData/'+plot_filename)



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
