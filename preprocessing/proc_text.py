from jamo import hangul_to_jamo
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import re
from params.hparams import hparams as hps
import numpy as np

PAD = '_'
EOS = '~'
SPACE = ' '

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS
symbols = PAD + SPACE + EOS + VALID_CHARS

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text):
    text_filter = "[,./!@#$%^&*()?]"
    text = re.sub(re.compile(text_filter), '', text)
    sequence = []
    if not 0x1100 <= ord(text[0]) <= 0x1113:
        text = ''.join(list(hangul_to_jamo(text)))
    for s in text:
        sequence.append(_symbol_to_id[s])
    sequence.append(_symbol_to_id['~'])
    # ~ 문장 구분자 추가\
    
    sequence = np.asarray(sequence)
    return sequence


def padding_sequences(sequences):
    max_text_sequence = hps.max_text_sequence
    padded_sequences = pad_sequences(
        sequences, maxlen=max_text_sequence, padding="post")
    return padded_sequences


def sequence_to_text(sequence):
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            result += s
    return result.replace('}{', ' ')
