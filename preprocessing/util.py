from preprocessing.proc_audio import get_mel_sp
from preprocessing.proc_text import text_to_sequence, padding_sequences
from params.hparams import hparams as hps
import os
import tqdm
import numpy as np

test = hps.test

text_name = 'transcript.v.1.4.test.txt' if test == True else 'transcript.v.1.4.txt'

def get_data():
    print('파일을 읽겠습니다.')
    token_list = []
    mel_list = []
    linear_list = []
    with open(os.path.join('in', text_name), encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().split('|')
            text = parts[2]
            
            sequence = text_to_sequence(text)
            wav_path = os.path.join('in', 'kss', parts[0])
            mel,spec = get_mel_sp(wav_path)
            
            token_list.append(sequence)
            mel_list.append(mel)
            linear_list.append(spec)
    print("파일을 다 읽었습니다.")
    return token_list, mel_list, linear_list

def padding_audio(self, audio_list, fill_value,last_dim, mel_input=False):
    middle_dim = 513 if mel_input else 512
    padded_audio = np.full(shape=(len(audio_list), middle_dim, last_dim),
                            fill_value=fill_value)

    for i, audio in enumerate(audio_list):
        padded_audio[i][:audio.shape[0]] = audio_list[i]

    return padded_audio
