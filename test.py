import numpy as np
from preprocessing.proc_text import text_to_sequence,padding_sequences
from librosa.feature.inverse import mel_to_audio
import argparse
import os
from tensorflow.keras.models import load_model
from params.hparams import hparams as hps
from librosa.output import write_wav
import re
from train import get_encoder_decoder, encoder_decoder_compile, get_decoder_num
import tensorflow as tf


def get_tained_model():
    num = get_decoder_num()
    encoder, decoder = get_encoder_decoder()

    decoder.load_weights("results/trained_model/decoder{}.ckpt".format(num-1))
    encoder.load_weights("results/trained_model/encoder{}.ckpt".format(num-1))

    encoder_decoder_compile(encoder,decoder)
    return encoder, decoder


def inference(text, encoder, decoder):
    processed_text = np.array(padding_sequences([text_to_sequence(text)]))

    encoded_text = encoder.predict(processed_text) 

    last_mel_outputs = np.zeros((1, 1, 80))
    stop = False
    n_iter = 0
    min_iter = hps.min_iter
    max_iter = hps.max_iter
    
    while not stop:
        n_iter += 1

        mel_input = np.reshape(last_mel_outputs, (1, -1, 80))

        pred = decoder.predict([encoded_text, mel_input])

        _, mel, linear, stop_token, weights = pred

        last_mel = mel[:,-1:]

        last_mel_outputs = np.append(last_mel_outputs, last_mel, axis=1)
        print('\n{}% 진행중입니다.'.format((100*n_iter)/max_iter))
        if (stop_token[0][-1] > 0.5 or n_iter >= max_iter) and n_iter > min_iter:
            mel_prediction = mel[0]
            linear_prediction = linear[0]
            alignments = np.transpose(weights[0])
            stop = True
        
    return mel_prediction, linear_prediction, alignments

def get_audio_file(mel_prediction):
    S = np.transpose(mel_prediction)
    audio = mel_to_audio(S,
                         sr = hps.sample_rate,
                         n_fft = hps.n_fft,
                         win_length = hps.win_length,
                         hop_length = hps.hop_length,
                         power = hps.power)

    write_wav("results/wav/inf.wav",audio,sr = hps.sample_rate)
    
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('text',type=str,help="어떤 말을 읽어드릴까요?(한글)")

    args = parser.parse_args()
    return args

def main():
    args = parser()
    text = args.text 
    print(text)

    encoder, decoder = get_tained_model()
    mel_prediction, linear_prediction, alignments =inference(text,encoder,decoder)
    print("예측이 끝났습니다.")
    print(linear_prediction)
    get_audio_file(mel_prediction)
    print("오디오 변환이 완료되었습니다. results/wav를 확인해보세요.")

if __name__ == "__main__":
    main()