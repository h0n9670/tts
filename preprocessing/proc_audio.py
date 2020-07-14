
import numpy as np
import librosa
from params.hparams import hparams as hps

def get_mel_linear(path):
    y, _ = librosa.core.load(
        path,
        sr=hps.sample_rate
    )
    # 잡음 제거
    mel = librosa.feature.melspectrogram(y, 
                                         sr=hps.sample_rate, 
                                         n_mels = hps.num_mels,
                                         n_fft=hps.n_fft,  
                                         power=hps.power)
    linear = librosa.stft(y, n_fft = hps.n_fft)

    mel = np.transpose(mel)
    linear = np.transpose(linear)

    return mel, linear

def get_audio(mel):
    s = np.transpose(mel)
    y = librosa.feature.inverse.mel_to_audio(s,
                                            sr = hps.sample_rate,
                                            n_fft=hps.n_fft,
                                            win_length = hps.win_length,
                                            hop_length = hps.hop_length,
                                            power = hps.power)

    return y


