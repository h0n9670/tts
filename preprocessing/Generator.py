from tensorflow.keras.utils import Sequence
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import os, sys
rootdir_name = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), '../params')
sys.path.append(rootdir_name)
from hparams import hparams as hps
from .proc_audio import get_mel_linear
from .proc_text import text_to_sequence,padding_sequences


class Generator(Sequence):
    def __init__(self,clas, number):
        super(Generator,self).__init__()
        self.batch_size = hps.batch_size
        self.num_mel = hps.num_mels
        self.clas = clas
        self.valid_rate = hps.valid_rate
        self.number = number
        self.text_name = 'transcript_{}.txt'.format(self.number)
        token_list, mel_list, linear_list = self.get_data()
        self.token_list = self._split_valid(token_list)
        self.mel_list = self._split_valid(mel_list)
        self.linear_list = self._split_valid(linear_list)
        self.data_size = len(self.token_list)
        self.on_epoch_end()
        self.num_mel = hps.num_mels
        self.num_freq = hps.num_freq
        self.outputs_per_step = hps.outputs_per_step
        
    def on_epoch_end(self):
        self.batches = np.arange(0, self.data_size, self.batch_size) 
    
        
    def __len__(self):
        return len(self.batches)   
    
    def __getitem__(self,batch_index):
        
        S_index, L_index = self._get_index(batch_index)
         
        token_input_list, mel_batch_list, linear_batch_list\
             = self._get_batch_list(S_index,L_index)
        
        zero_mel = np.zeros((1, self.num_mel))
        
        mel_input_list = []
        stop_batch_list = []
        for i in range(0, L_index - S_index):
            target_mel = mel_batch_list[i]

            mel_input = np.append(zero_mel, target_mel, axis=0)
            stop = np.full((len(mel_batch_list[i]), 1),fill_value=0.)
            stop[-1, 0] = 1.

            mel_input_list.append(mel_input)
            stop_batch_list.append(stop)

        mel_input = self.padding_audio(mel_input_list, 0.,self.num_mel, True)
        mel_batch = self.padding_audio(mel_batch_list, 0., self.num_mel)
        linear_batch = self.padding_audio(linear_batch_list, 0., self.num_freq+1)
        stop_batch = self.padding_audio(stop_batch_list, 1., 1)
                
        token_input = np.array(token_input_list,dtype=np.float32)
        
        # print("token input type :", type(token_input))
        # print("token input sample :", type(token_input[0]))
        # print("token input sample :", token_input[0])
        # print("mel input type :", type(mel_input))
        # print("mel input sample :", type(mel_input[0]))
        # print("mel input sample :", mel_input[0])
        # print("mel batch type :", type(mel_batch))
        # print("mel_batch sample :", type(mel_batch[0]))
        # print("mel_batch sample :", mel_batch[0])
        # print("linear_batch type :", type(linear_batch))
        # print("linear_batch sample :", type(linear_batch[0]))
        # print("linear_batch sample :", linear_batch[0])
        # print("stop_batch type :", type(stop_batch))
        # print("stop_batch sample :", type(stop_batch[0]))
        # print("stop_batch sample :", stop_batch[0])
        mel_input = mel_input[:,:-1]

        return [token_input, mel_input], [mel_batch, mel_batch,linear_batch, stop_batch]
        

    def _get_index(self,batch_index):
        start_index_list = np.arange(0, self.data_size, self.batch_size)
        last_index = self.data_size - (self.data_size % self.batch_size)

        np.random.shuffle(start_index_list)

        if start_index_list[batch_index] == last_index:
            S_index = start_index_list[batch_index]
            L_index = start_index_list[batch_index] + \
                (self.data_size % self.batch_size)
        else:
            S_index = start_index_list[batch_index]
            L_index = S_index + self.batch_size
        print("S_index : ", S_index)
        print("L_index : ", L_index)
        print("batch_index : ", batch_index)
        print("batch_size : ", self.batch_size)
        print("data_size : ", self.data_size)
        
        S_index = int(S_index)
        L_index = int(L_index)

        return S_index, L_index
    
    def _split_valid(self,data_list):
        a, b = train_test_split(data_list, 
                                test_size = self.valid_rate, shuffle=False)
        return b if self.clas == "valid" else a
        
    
    def _get_batch_list(self, S_index, L_index):      
        
        token_input_list = self.token_list[S_index: L_index]
        mel_batch_list = self.mel_list[S_index: L_index]
        linear_batch_list = self.linear_list[S_index: L_index]
        
        return token_input_list, mel_batch_list, linear_batch_list

    
    def get_data(self):
        print('\n{} 파일을 읽겠습니다.'.format(self.clas))
        token_list = []
        mel_list = []
        linear_list = []
        with open(os.path.join('in', self.text_name), encoding='utf-8') as f:
            for line in tqdm(f):
                parts = line.strip().split('|')
                text = parts[2]
                
                sequence = text_to_sequence(text)
                wav_path = os.path.join('in', 'kss', parts[0])
                mel,linear = get_mel_linear(wav_path)
                
                token_list.append(sequence)
                mel_list.append(mel)
                linear_list.append(linear)
        token_list = padding_sequences(token_list)
        print("\n{} 파일을 다 읽었습니다.".format(self.clas))
        return token_list, mel_list, linear_list


    def padding_audio(self,audio_list, fill_value, last_dim, mel_input=False):
        middle_dim = 513 if mel_input else 512
        padded_audio = np.full(shape=(len(audio_list), middle_dim, last_dim),
                            fill_value=fill_value,dtype=np.float32)

        for i, audio in enumerate(audio_list):
            padded_audio[i][:audio.shape[0]] = audio_list[i]

        return padded_audio

    
if __name__ == "__main__":
    # generator = Generator()
    # a = generator.__getitem__(0)
    # print(np.array(a[0]).shape)
    pass
