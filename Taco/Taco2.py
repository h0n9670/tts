from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Concatenate,\
    Reshape, Add, Lambda, Dense, Masking
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
import tensorflow as tf

import os
import sys
rootdir_name_param = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), '../params')
sys.path.append(rootdir_name_param)
rootdir_name_taco = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), '../Taco')
sys.path.append(rootdir_name_taco)
from hparams import hparams as hps

from Attention import LocationSensitiveAttentionLayer
from Layers import Conv1d_3, PreNet, PostNet, CBHG


def Encoder(): 

    inputs = Input(shape=(None,),name="encoder_inputs")

    masking = Masking(mask_value=0,name = "Encoder_masking")
        
    embedding = Embedding(input_dim=hps.n_symbols,
                          output_dim=hps.embedding_dim,
                          name="Character_embedding")

    bidlstm = Bidirectional(LSTM(units=hps.lstm_size,
                                 recurrent_dropout=hps.lstm_drop_rate,
                                 return_sequences=True),
                            name='Bidir_LSTM')
    conv1d_3 = Conv1d_3()
 
    x = masking(inputs)
    # making 대신 embedding masking zero 도 있는데 확인은 안해보았다.
    x = embedding(x)
    x = conv1d_3(x)
    outputs = bidlstm(x)
    
    return Model(inputs,x,name="encoder")


def Decoder():

    prenet = PreNet()
    attention = LocationSensitiveAttentionLayer()
    postnet = PostNet()
    cbhg = CBHG()
                  
    encoded = Input(shape=(None, 2*hps.lstm_size), name="Encoded_Text")
    decoding = Input(shape=(None,80), name="Decoding")
    
    masked_decoding = Masking(mask_value=0.,name = "Decoder_masking")(decoding)

    prenet_out = prenet(masked_decoding)
    context, alignments = attention([encoded, prenet_out])
    apply_weight = Concatenate(axis=-1)([prenet_out, context])

    decoding_out = Dense(hps.Frame_shape, name = "Frame_projection")(apply_weight)
    stop = Dense(hps.Stop_shape, name='StopProjection')(apply_weight)
    decoder_out = Reshape((-1, hps.num_mels))(decoding_out)
    stop = Reshape((-1, 1))(stop)

    residual = postnet(decoder_out)
    projected_residual = Dense(
        hps.projection_shape, name='residual_projection')(residual)
    mel_out = Add(name="mel_predictions")(
        [decoder_out, projected_residual])
    
    post_outputs = cbhg(mel_out)
    linear_out = Dense(hps.num_freq+1, name='linear_projection')(post_outputs)

    decoder_model = Model([encoded, decoding],
                          [decoder_out, mel_out, linear_out, stop, alignments],
                           name = "decoder")

    return decoder_model


    
class Taco2:
    def __init__(self, encoder, decoder):
        
        self.encoder = encoder
        self.decoder = decoder
  
    def __call__(self):
        
        encoding = Input(shape=(None,), name='encoder_input_train_model')
        
        decoding = Input(shape=(None, hps.num_mels), 
                         name="Decoder_input_train_moel")
        encoder_out = self.encoder(encoding)
        decoding_out, mel_out, linear_out, stop, _ = self.decoder(
            [encoder_out, decoding])

        self.Taco2 = Model(inputs=[encoding, decoding],
                           outputs=[decoding_out, mel_out, linear_out, stop])
        return self.Taco2


if __name__ == "__main__":
    encoder = Encoder()
    # decoder = Decoder()
    # taco2 = Taco2(encoder, decoder)

    plot_model(encoder, to_file="Taco/png/encoder.png", show_shapes=True)
    # plot_model(decoder, to_file="Taco/png/decoder.png", show_shapes=True)
    # plot_model(taco2, to_file="Taco/png/taco2.png", show_shapes=True)
    # pass
