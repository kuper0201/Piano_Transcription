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

train_x, train_onset = [], []
valid_x, valid_onset = [], []

model = None

def dataSetGenerator(x_path, onset_path):
    for a, b in zip(x_path, onset_path):
        X, ONSET = [], []
        X.append(np.load(a))
        ONSET.append(np.load(b))

        X = np.concatenate(X, axis=0)
        ONSET = np.concatenate(ONSET, axis=0)

        X = X / np.max(X)

        model.reset_states()
        for x, onset in zip(X, ONSET):
            yield (x, onset)

def buildModel():
    input_layer = Input(batch_input_shape=(10, 100, 264), name='onset_input')

    # Onset LSTM
    onset_lstm = Bidirectional(LSTM(128, activation='tanh', return_sequences=True, stateful=True, name='onset_lstm'))(input_layer)

    # Output
    onset_out = TimeDistributed(Dense(88, activation='sigmoid', kernel_initializer='he_normal', name='onset_output'))(onset_lstm)

    model = keras.Model(inputs=input_layer, outputs=onset_out)
    return model

def train(trainX, trainOnset, validX, validOnset):
    for x_name, onset_name in zip(glob.glob(trainX), glob.glob(trainOnset)):
        train_x.append(x_name)
        train_onset.append(onset_name)

    for x_name, onset_name in zip(glob.glob(validX), glob.glob(validOnset)):
        valid_x.append(x_name)
        valid_onset.append(onset_name)

    input_signature = (tensorflow.float32, tensorflow.int8)
    in_out_shape = ([100, 264], [100, 88])

    global model
    model = buildModel()
    trainSet = tensorflow.data.Dataset.from_generator(dataSetGenerator, input_signature, in_out_shape, args=[train_x, train_onset])
    validSet = tensorflow.data.Dataset.from_generator(dataSetGenerator, input_signature, in_out_shape, args=[valid_x, valid_onset])
    trainSet = trainSet.batch(10, drop_remainder=True).prefetch(tensorflow.data.experimental.AUTOTUNE)
    validSet = validSet.batch(10, drop_remainder=True).prefetch(tensorflow.data.experimental.AUTOTUNE)

    checkpoint = ModelCheckpoint('onset_detector.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stop = EarlyStopping(patience=5, monitor='val_loss', verbose=1, mode='auto')

    opt = tensorflow.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(trainSet, validation_data=validSet, epochs=1000, shuffle=False, callbacks=[checkpoint, early_stop])
    model.save('onset_last.h5')

if __name__ == '__main__':
    train('../PreProc/trainX/*.npy', '../PreProc/trainONSET/*.npy', '../PreProc/validX/*.npy', '../PreProc/validONSET/*.npy')