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
