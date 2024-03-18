import glob
import numpy as np
from keras import Model
from tensorflow import keras
from keras.layers import Rescaling, Dense, Dropout, Input, LayerNormalization, Activation, Bidirectional, LSTM, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.python.keras.optimizers

import warnings
warnings.filterwarnings("ignore")

import tensorflow.python.keras.mixed_precision.policy as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

train_x, train_offset = [], []
valid_x, valid_offset = [], []

model = None

def dataSetGenerator(x_path, offset_path):
    for a, b in zip(x_path, offset_path):
        X, OFFSET = [], []
        X.append(np.load(a))
        OFFSET.append(np.load(b))

        X = np.concatenate(X, axis=0)
        OFFSET = np.concatenate(OFFSET, axis=0)

        X = X / np.max(X)

        model.reset_states()
        for x, offset in zip(X, OFFSET):
            yield (x, offset)

def buildModel():
    input_layer = Input(batch_input_shape=(10, 100, 264), name='len_input')
    
    # Offset LSTM
    offset_lstm = Bidirectional(LSTM(128, activation='tanh', return_sequences=True, stateful=True, name='len_lstm'))(input_layer)

    # Output
    offset_out = TimeDistributed(Dense(88, activation='sigmoid', kernel_initializer='he_normal', name='len_output'))(offset_lstm)

    model = keras.Model(inputs=input_layer, outputs=offset_out)
    return model

def train(trainX, trainOffset, validX, validOffset):
    for x_name, offset_name in zip(glob.glob(trainX), glob.glob(trainOffset)):
        train_x.append(x_name)
        train_offset.append(offset_name)

    for x_name, offset_name in zip(glob.glob(validX), glob.glob(validOffset)):
        valid_x.append(x_name)
        valid_offset.append(offset_name)

    input_signature = (tensorflow.float32, tensorflow.int8)
    in_out_shape = ([100, 264], [100, 88])

    global model
    model = buildModel()
    trainSet = tensorflow.data.Dataset.from_generator(dataSetGenerator, input_signature, in_out_shape, args=[train_x, train_offset])
    validSet = tensorflow.data.Dataset.from_generator(dataSetGenerator, input_signature, in_out_shape, args=[valid_x, valid_offset])
    trainSet = trainSet.batch(10, drop_remainder=True).prefetch(tensorflow.data.experimental.AUTOTUNE)
    validSet = validSet.batch(10, drop_remainder=True).prefetch(tensorflow.data.experimental.AUTOTUNE)

    checkpoint = ModelCheckpoint('offset_detector.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stop = EarlyStopping(patience=5, monitor='val_loss', verbose=1, mode='auto')

    opt = tensorflow.optimizers.Adam(learning_rate=0.002)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(trainSet, validation_data=validSet, epochs=1000, shuffle=False, callbacks=[checkpoint, early_stop])

if __name__ == '__main__':
    train('../PreProc/trainX/*.npy', '../PreProc/trainOFFSET/*.npy', '../PreProc/validX/*.npy', '../PreProc/validOFFSET/*.npy')
