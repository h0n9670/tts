'''
conv1d_3, prenet, postnet, convBank, Highwaynet, CBHG
'''
from tensorflow.keras.layers import Conv1D, BatchNormalization, \
    LeakyReLU,Dropout, Dense, Layer, Conv1D, Activation, ReLU, Bidirectional, LSTM,\
    MaxPool1D
from tensorflow.keras.models import Sequential
import os
import sys
rootdir_name = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), '../params')
sys.path.append(rootdir_name)
from hparams import hparams as hps

class Conv1d_3(Layer):
    '''encoder에 들어가는 conv_3'''
    
    def __init__(self):
        super(Conv1d_3, self).__init__()

        self.conv1d = Conv1D(kernel_size=hps.conv3_kernel_size,
                                 filters=hps.conv3_filters,
                                 strides=hps.conv3_strides,
                                 padding='same',
                                 use_bias=False)
        self.bn = BatchNormalization()
        self.lr = LeakyReLU(0.01)
        self.do = Dropout(hps.conv3_drop_rate)
            
    def call(self, inputs, training=False):
        x = inputs
        for _ in range(3):
            x = self.conv1d(x)
            x = self.bn(x)
            x = self.lr(x)
            x = self.do(x)
        return x
        


class PreNet(Layer):
    '''attention 들어가기 전에 mel형식의 데이터를 입력'''
    def __init__(self):
        super(PreNet, self).__init__()
        
        self.sizes = hps.PreNet_layers_sizes
        self.dropout_rate = hps.PreNet_drop_rate
        self.prenet = Sequential()
        
    def __call__(self, inputs):
        for i,size in enumerate(self.sizes):
            self.prenet.add(Dense(
                units=size,
                activation='relu',
                name = "prenet_dence_{}".format(i)
                ))
            self.prenet.add(Dropout(rate=self.dropout_rate,
                                    name = "prenet_dropout_{}".format(i)
                                    ))
        prenet_result = self.prenet(inputs)
        return prenet_result

    

class PostNet(Layer):
    '''decoder 결과를 mel로 변환시 '''
    def __init__(self):
        super(PostNet, self).__init__()
        
        self.filters = hps.PostNet_filters
        self.kernel_size = hps.PostNet_kernel_size
        self.stride = hps.PostNet_stride
        self.dropout = hps.PostNet_drop_rate

        self.postnet = Sequential()

    def __call__(self, inputs):
        for _ in range(0, 4):
            a = Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding='same'
                )
            self.postnet.add(a)

            self.postnet.add(BatchNormalization())

            self.postnet.add(Activation('tanh'))

            self.postnet.add(Dropout(rate=self.dropout))
        return self.postnet(inputs)

########################################################
class ConvBank(Layer):
    def __init__(self):
        super(ConvBank, self).__init__()

        self.stack_count = hps.ConvBank_size
        self.filters = hps.ConvBank_filters

    def build(self, input_shapes):
        self.convBank = Sequential()
        for i in range(self.stack_count):
            self.convBank.add(Conv1D(
                filters=self.filters,
                kernel_size=i + 1,
                padding='same',
                use_bias=False
            ))
            self.convBank.add(BatchNormalization())
            self.convBank.add(ReLU())

    def call(self, inputs):
        return self.convBank(inputs)

class Highwaynet(Layer):
    def __init__(self, size):
        super(Highwaynet, self).__init__()
        self.relu_layer = Dense(
            units=size,
            activation='relu'
        )
        self.sigmoid_layer = Dense(
            units=size,
            activation='sigmoid'
        )

    def call(self, inputs):
        relu_out = self.relu_layer(inputs)
        sig_out = self.sigmoid_layer(inputs)

        return relu_out * sig_out + (1.0 - sig_out)


class CBHG(Layer):
    '''convbank -> max_pooling -> conv1D_preojection -> Residual concat -> 
    bid rnn'''
    def __init__(self):
        self.pool_size = hps.Pool_size
        self.pool_strides = hps.Pool_strides
        
        self.project_conv_filters = hps.project_conv_filters
        self.project_conv_kernel_size = hps.project_conv_kernel_size
        
        self.highwaynet_count = hps.HighwayNet_count
        self.highwaynet_size = hps.HighwayNet_size
        
        self.rnn_size = hps.RNN_size
        self.rnn_zoneout_rate = hps.RNN_zoneout_rate

        super(CBHG, self).__init__()

    def build(self, input_shapes):

        self.convBank = ConvBank()

        self.max_pooling = MaxPool1D(
            pool_size = self.pool_size,
            strides = self.pool_strides,
            padding = 'same',
            name = "MaxPooling"
        )

        self.conv1D_Projection = Sequential(name="conv1D_Projection")
        for i in range(0, 2):
            self.conv1D_Projection.add(Conv1D(
                filters=self.project_conv_filters,
                kernel_size=self.project_conv_kernel_size,
                padding='same',
                use_bias=False
            ))

            self.conv1D_Projection.add(BatchNormalization())
            if i == 0 :
                self.conv1D_Projection.add(ReLU())


        if input_shapes[-1] != self.project_conv_filters:
            d = Dense(
                units=input_shapes[-1]
            )

        self.highwaynet = Sequential(name = "HighwayNet")
        if input_shapes[-1] != self.highwaynet_size:
            self.highwaynet.add(Dense(
                units=self.highwaynet_size
                ))

        for _ in range(self.highwaynet_count):
            self.highwaynet.add(Highwaynet(
                size=self.highwaynet_size
                ))

        self.rnn_layer = Bidirectional(LSTM(
            units=self.rnn_size,
            recurrent_dropout=self.rnn_zoneout_rate,
            return_sequences=True,
            name = "Bidrectional_lstm"
        ))

    def call(self, inputs):
        
        x = self.convBank(inputs)
        x = self.max_pooling(x)
        x = self.conv1D_Projection(x)
        x = x + inputs    # Residual
        x = self.highwaynet(x)
        return self.rnn_layer(x)
