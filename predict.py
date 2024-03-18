from os.path import split, join

import matplotlib.pyplot as plt
import pretty_midi
import seaborn as sns
from pretty_midi import Note
import numpy as np
import librosa
from pydub import AudioSegment
from tensorflow import keras
import tensorflow.python.keras.mixed_precision.policy as mixed_precision
import warnings
from librosa.feature import rms

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

warnings.filterwarnings("ignore")
AudioSegment.converter = 'ffmpeg'

def one_to_midi(notes, offsets, fileName, time):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=1)

    notes = notes.T
    offsets = offsets.T
    for pitch, hor in enumerate(notes):
        nz = np.where(hor != 0)[0]
        if len(nz) == 0:
            continue

        visit = [False] * len(hor)
        off = offsets[pitch]
        for idx in nz:
            i = idx
            while i < len(off) and off[i] != 0:
                visit[i] = True
                i += 1

        idx = 0
        while idx < len(visit):
            startTime = idx * time
            endTime = startTime

            while idx < len(visit) and visit[idx] == True:
                endTime += time
                idx += 1

            if startTime != endTime:
                instrument.notes.append(Note(velocity=100, pitch=pitch + 21, start=startTime, end=endTime))
            idx += 1

    print('saving...')
    pm.instruments.append(instrument)
    pm.write(fileName)
    print('save complete')

def test(X_test_path):
    len_model = keras.models.load_model("models/offset_detector.h5")
    onset_model = keras.models.load_model("models/onset_detector.h5")
    print('model loaded')

    y, sr = librosa.load(X_test_path, sr=16000)
    cqt = librosa.cqt(y, sr=sr, fmin=librosa.midi_to_hz(21), n_bins=264, hop_length=160, bins_per_octave=36)
    cqt = np.abs(cqt)
    cqt = cqt.T

    cqt = cqt / np.max(cqt)

    # 시퀀스 길이 / 배치 사이즈 상수
    one_seq = 100
    batch_size = 10

    # 시퀀스 패딩
    pad_size = one_seq - (cqt.shape[0] % one_seq)
    cqt = np.pad(cqt, ((0, pad_size), (0, 0)), mode='constant')
    cqts = cqt.reshape(cqt.shape[0] // one_seq, one_seq, 264)

    # Batch 패딩
    desired_shape = (batch_size * ((cqts.shape[0] + (batch_size - 1)) // batch_size), one_seq, 264)
    padding_shape = (desired_shape[0] - cqts.shape[0], desired_shape[1] - cqts.shape[1], desired_shape[2] - cqts.shape[2])
    cqts = np.pad(cqts, ((0, padding_shape[0]), (0, padding_shape[1]), (0, padding_shape[2])), mode='constant')

    # 데이터 예측
    len_model.reset_states()
    len_result = len_model.predict(cqts, batch_size=batch_size)
    onset_result = onset_model.predict(cqts, batch_size=batch_size)

    onset = onset_result.reshape(onset_result.shape[0] * one_seq, 88)
    offset = len_result.reshape(len_result.shape[0] * one_seq, 88)

    #onset = len_result[0].reshape(len_result[0].shape[0] * one_seq, 88)
    #offset = len_result[1].reshape(len_result[1].shape[0] * one_seq, 88)

    onset = np.where(onset >= 0.5, 1, 0)
    offset = np.where(offset >= 0.3, 1, 0)

    to_elapse = librosa.frames_to_time(1, sr=sr, hop_length=160)
    one_to_midi(notes=onset, offsets=offset, fileName=join('mid/', split(X_test_path)[-1][:-4] + '.mid'), time=to_elapse)

if __name__ == '__main__':
    test('../../LSTMTest/fd.wav')
