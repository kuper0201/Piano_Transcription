import glob
from os import makedirs
from os.path import join, exists, split
from multiprocessing import Process

import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import note_seq
import numpy as np

MIDI_paths, MP3_paths = [], []
VAL_MIDI_paths, VAL_MP3_paths = [], []

def preprocess_wave_and_midi(MIDI_path, x_path, x_save_path, onset_save_path, offsets_save_path):
    # 파일 로드, CQT 변환
    y, sr = librosa.load(x_path, sr=16000)
    cqt = librosa.cqt(y, sr=sr, fmin=librosa.midi_to_hz(21), n_bins=264, hop_length=160, bins_per_octave=36)
    cqt = np.abs(cqt)
    cqt = cqt.T

    cqt = np.pad(cqt, ((0, 10), (0, 0)), mode='constant')

    # 빈 결과 배열 생성
    onset = np.zeros((cqt.shape[0], 88))
    offset = np.zeros((cqt.shape[0], 88))

    # 패딩
    one_seq = 100
    pad_size = one_seq - (cqt.shape[0] % one_seq)
    cqt = np.pad(cqt, ((0, pad_size), (0, 0)), mode='constant')
    onset = np.pad(onset, ((0, pad_size), (0, 0)), mode='constant')
    offset = np.pad(offset, ((0, pad_size), (0, 0)), mode='constant')

    # MIDI 노트 순회
    ns = note_seq.midi_file_to_sequence_proto(MIDI_path)
    for i in ns.notes:
        note = i.pitch - 21

        # 온셋 + 길이 시퀀스
        for x in range(int(i.start_time * 100), int(i.start_time * 100) + 4):
            onset[x, note] = 1

        # for x in range(int(i.start_time * 100) + 4, int(i.end_time * 100)):
        #     onset[x, note] = 1

        # 오프셋 시퀀스
        #for x in range(int(i.end_time * 100) - 2, int(i.end_time * 100) + 3):
        for x in range(int(i.start_time * 100), int(i.end_time * 100)):
            offset[x, note] = 1

        # 길이 감지
        #offset[int(i.start_time * 100), note] = i.end_time - i.start_time

        # 온셋 감지
        # onset[int(i.start_time * 100), note] = 1

    # 크기 변환
    cqts = cqt.reshape(cqt.shape[0] // one_seq, one_seq, 264)
    onsets = onset.reshape(onset.shape[0] // one_seq, one_seq, 88)
    offsets = offset.reshape(offset.shape[0] // one_seq, one_seq, 88)

    #onsets = []
    #for oh in onset:
    #    onsets.append(np.logical_or.reduce(oh, axis=0).astype(int))

    #onsets = np.array(onsets)
    #onsets = onsets.reshape(onsets.shape[0], 1, 88)

    name = split(x_path)[-1][:-4]
    np.save(join(x_save_path, name), cqts)
    np.save(join(onset_save_path, name), onsets)
    np.save(join(offsets_save_path, name), offsets)

def createDirectory(directory):
    if not exists(directory):
        makedirs(directory)

def multi(MIDI, MP3, saveX, saveONSET, saveOFFSET):
    for a, b in zip(MIDI, MP3):
        preprocess_wave_and_midi(a, b, saveX, saveONSET, saveOFFSET)

def main(midi_dir, valid_midi_dir, save_path):
    createDirectory(join(save_path, 'trainX'))
    createDirectory(join(save_path, 'validX'))
    createDirectory(join(save_path, 'trainONSET'))
    createDirectory(join(save_path, 'validONSET'))
    createDirectory(join(save_path, 'trainOFFSET'))
    createDirectory(join(save_path, 'validOFFSET'))

    for name in glob.glob(midi_dir):
        MIDI_paths.append(name)
        MP3_paths.append(name[:-4] + "wav")

    for name in glob.glob(valid_midi_dir):
        VAL_MIDI_paths.append(name)
        VAL_MP3_paths.append(name[:-4] + "wav")

    train_len = len(MIDI_paths) // 4
    Process(target=multi, args=(MIDI_paths[:train_len], MP3_paths[:train_len], join(save_path, 'trainX'), join(save_path, 'trainONSET'), join(save_path, 'trainOFFSET'))).start()
    Process(target=multi, args=(MIDI_paths[train_len:2 * train_len], MP3_paths[train_len:2 * train_len], join(save_path, 'trainX'), join(save_path, 'trainONSET'), join(save_path, 'trainOFFSET'))).start()
    Process(target=multi, args=(MIDI_paths[2 * train_len:3 * train_len], MP3_paths[2 * train_len:3 * train_len], join(save_path, 'trainX'), join(save_path, 'trainONSET'), join(save_path, 'trainOFFSET'))).start()
    Process(target=multi, args=(MIDI_paths[3 * train_len:], MP3_paths[3 * train_len:], join(save_path, 'trainX'), join(save_path, 'trainONSET'), join(save_path, 'trainOFFSET'))).start()

    valid_len = len(VAL_MIDI_paths) // 2
    Process(target=multi, args=(VAL_MIDI_paths[:valid_len], VAL_MP3_paths[:valid_len], join(save_path, 'validX'), join(save_path, 'validONSET'), join(save_path, 'validOFFSET'))).start()
    Process(target=multi, args=(VAL_MIDI_paths[valid_len:], VAL_MP3_paths[valid_len:], join(save_path, 'validX'), join(save_path, 'validONSET'), join(save_path, 'validOFFSET'))).start()

if __name__ == '__main__':
    #main('../../LSTMData/train/*.midi', '../../LSTMData/valid/*.midi', '../PreProc')
    main('../Data/train/*.midi', '../Data/valid/*.midi', '../PreProc')