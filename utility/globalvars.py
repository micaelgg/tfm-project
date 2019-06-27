"""
Base code:
https://github.com/RayanWang/Speech_emotion_recognition_BLSTM/blob/master/utility/globalvars.py
"""
globalVar = 0
dataset = 'ravdess'
emotions = []

nb_attention_param = 256
attention_init_value = 1.0 / 256
max_len = 1024
nb_classes = 8

masking_value = -100.0

nb_features = 36
nb_hidden_units = 512   # number of hidden layer units
dropout_rate = 0.5
nb_lstm_cells = 128

label_features = [
    'zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread',
    'spectral_entropy', 'spectral_flux', 'spectral_rolloff', 'mfcc_1',
    'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8',
    'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1',
    'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7',
    'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12',
    'chroma_std', 'harmonic_ratio', 'pitch'
]
