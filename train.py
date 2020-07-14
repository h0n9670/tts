from Taco.Taco2 import Encoder, Decoder, Taco2
from preprocessing.Generator import Generator
import tensorflow.keras.backend as K
from params.hparams import hparams as hps
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os
from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.optimizers import Adam
import re

def get_encoder_decoder():        
    encoder = Encoder()
    decoder = Decoder()
    return encoder, decoder

def encoder_decoder_compile(encoder,decoder):
    encoder.compile(optimizer= Adam(0.01), loss='mse', metrics=['accuracy'])
    decoder.compile(optimizer= Adam(0.01), loss='mse', metrics=['accuracy'])

def get_taco2(encoder, decoder):
    taco2 = Taco2(encoder,decoder)
    taco2 = taco2()
    return taco2

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('number',
                        type=int,
                        help="학습할 데이터 번호를 입력하세요.",
                        default = 0)
    args = parser.parse_args()
    return args

def get_generator():
    args = parser()
    print(args.number)
    # train_generator = Generator("train")
    train_generator = Generator("train",args.number)
    # valid_generator = Generator('valid')
    valid_generator = Generator('valid',args.number)
    return train_generator,valid_generator

def stop_loss(y_true, y_pred):
    x = K.mean(K.binary_crossentropy(y_true, y_pred))
    return x

def linear_loss(y_true, y_pred):
    l1 = K.abs(y_true - y_pred)
    n_priority_freq = int(2000 / (hps.sample_rate * 0.5) * hps.num_freq)
    x = 0.5 * K.mean(l1) + 0.5 * K.mean(l1[:,:,0:n_priority_freq])
    return x

def get_callbacks():
    callbacks = []
    callbacks.append(EarlyStopping(monitor = hps.early_monitor,
                                   min_delta = hps.early_min_delta,
                                   patience = hps.early_patience,
                                   mode = hps.early_mode))

    callbacks.append(TensorBoard("results/tensorboard",
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=True))
    
    return callbacks

def confirm_model_existence():
    abspath = os.path.abspath('results')
    filelist = os.listdir(abspath)
    decoderpath = os.path.join(abspath,"trained_model","decoder{}.tf".format(args.number-1))
    confirm_model = os.path.isdir(decoderpath)
    return confirm_model

def get_decoder_num():
    abspath = os.path.abspath('results/trained_model')
    filelist = os.listdir(abspath)
    decoder_list = [file for file in filelist if file.startswith('decoder')]

    num_list = [int(re.findall('\d',file)[0]) for file in decoder_list]
    num_list = list(set(num_list))
    if len(num_list) == 0:
        decoder_num = 0
    elif len(num_list)==1:
        decoder_num = num_list[0]+1
    else :
        decoder_num = max(num_list)+1
    return decoder_num
    


def main():
    decoder_num = get_decoder_num()
    
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        encoder, decoder = get_encoder_decoder()
        if decoder_num != 0 :
            decoder.load_weights("results/trained_model/decoder{}.ckpt".format(decoder_num-1))
            encoder.load_weights("results/trained_model/encoder{}.cktp".format(decoder_num-1))

        encoder_decoder_compile(encoder,decoder)
        
        taco2 = get_taco2(encoder,decoder)
        loss = ["mse","mse",linear_loss,stop_loss]
        taco2.compile(optimizer = Adam(0.01),loss = loss ,metrics=['accuracy'])
        # 사용자 정의 함수까지 저장이 안되서 재 compile

    train_generator, valid_generator = get_generator() 
    callbacks = get_callbacks()

    history = taco2.fit_generator(generator = train_generator,
                                  epochs = hps.epochs,
                                  validation_data = valid_generator,
                                  callbacks = callbacks
                                  )
    args = parser()                              
    encoder.save_weights("results/trained_model/encoder{}.ckpt".format(decoder_num))
    decoder.save_weights("results/trained_model/decoder{}.ckpt".format(decoder_num))

    print("모델 학습이 완료되었습니다.results/trained_model을 확인하세요.")
if __name__ == "__main__":
    main()
