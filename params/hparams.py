class hparams:

##################################################
########    Audio    #############################
##################################################
    num_mels = 80
    num_freq = 1024
    sample_rate = 16000
    frame_length = 1024
    frame_shift = 256
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20
    power = 1.5
    gl_iters = 100
    max_abs_mel = 4
    mel_dim = 80
    n_fft = 2048
    win_length = 800  # 0.05 * sr
    hop_length = 200   # 0.0125* sr
    
##################################################
########## text ##################################
##################################################

    n_symbols = 70  # 한글자모 +space + ~ + _
    max_text_sequence = 100


 ################################################
 ########## layers ##############################
 ################################################
    '''
    Encoder
        - Conv 3
        - lstm
    '''
    
    # embedding
    embedding_dim = 512
    
    # conv3
    conv3_kernel_size = 5
    conv3_filters = 512
    conv3_strides = 1
    conv3_drop_rate = 0.5
    
    # lstm
    lstm_size = 256
    lstm_drop_rate = 0.5

    '''
    Decoder
        - PreNet
        - Attention
            - Decoderlstm
                - ZoneoutLstmCell
            
        - frame_projection
        - stop_projection
    '''
    # PreNet
    PreNet_layers_sizes = [256, 256]
    PreNet_drop_rate = 0.5
    PreNet_activation = "relu"

    # Decoderlstm
    DecoderRNN_layers = 2
    DecoderRNN_size = 1024
    DecoderRNN_zoneout = 0.1

    # Attention
    LSA_dim = 128
    LSA_filters = 32
    LSA_kernel = (31,)

    # Decoder
    divided = 2
    
    # FrameProjection
    Frame_shape = 80
    
    # StopProjection
    Stop_shape = 1
    Stop_activation = 'sigmoid'
    
    '''
    get mel linear form
        - PostNet
        - CBHG
            -Highwaynet
            -ConvBank
    '''
        
    # PostNet
    PostNet_filters = 512
    PostNet_kernel_size = 5
    PostNet_stride = 1
    PostNet_drop_rate = 0.0
    
    # projection
    projection_shape = 80
    
    # ConvBank
    ConvBank_size = 8
    ConvBank_filters = 256
    
    # HighwayNet
    HighwayNet_count = 4
    HighwayNet_size = 128
    
    # CBHG
    Pool_size = 2
    Pool_strides = 1
    
    project_conv_filters = 80
    project_conv_kernel_size = 3
    
    RNN_size = 256
    RNN_zoneout_rate = 0.0

    ##################################################################
    ######### train ##################################################
    ##################################################################
    
    '''
    generator   
    '''
    batch_size = 32
    outputs_per_step = 1
    '''
    '''
    valid_rate = 0.1
    epochs = 10
    '''
    earlystopping
    '''
    early_monitor = "val_loss"
    early_min_delta = 0.001
    early_patience = 3
    early_mode = 'min'

    '''
    checkpoint
    '''
    check_monitor = 'val_loss'
    check_mode = "min"
    check_period = 1

    '''
    tensorboard
    '''
    histogram_freq = 0
    
    ##################################################################
    ######### test  ##################################################
    ##################################################################
    
    min_iter = 20
    max_iter = 400