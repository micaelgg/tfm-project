"""
Base code:
https://github.com/RayanWang/Speech_emotion_recognition_BLSTM/blob/master/dataset.py
"""
import itertools
import os
import librosa
import numpy as np


class Dataset:
    """
    Emotion identifiers
    0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral', 7: 'calm', 8: 'boredom'
    """

    def __init__(self, path, name_dataset, gender, emotions, number_emo, frame_size, step):
        self.name_dataset = name_dataset
        self.name_emotions = emotions
        self.emotions = number_emo
        self.frame_size = frame_size
        self.step = step
        self.dictionary = dict(zip(np.arange(0, len(emotions)), emotions))
        self.targets = []
        self.data = []
        self.subjects = []
        self.subjects_gender = []

        if name_dataset == "enterface":
            self.classes = {0: 'an', 1: 'di', 2: 'fe', 3: 'ha', 4: 'sa', 5: 'su'}
            self.get_enterface05_dataset(path, gender, number_emo)
        elif name_dataset == "berlin":
            self.classes = {0: 'W', 1: 'E', 2: 'A', 3: 'F', 4: 'T', 8: 'L', 6: 'N'}
            self.get_berlin_dataset(path, gender, number_emo)
        elif name_dataset == "ravdess":
            self.classes = {0: '05', 1: '07', 2: '06', 3: '03', 4: '04', 5: '08', 6: '01', 7: '02'}
            self.get_ravdess_dataset(path, gender, number_emo)
        elif name_dataset == "cremad":
            self.classes = {0: 'ANG', 1: 'DIS', 2: 'FEA', 3: 'HAP', 4: 'SAD', 6: 'NEU'}
            self.get_cremad_dataset(path, gender, number_emo)

    def get_ravdess_dataset(self, path, gender, emotions):
        """name format = 02 - 01 - 06 - 01 - 02 - 01 - 12
        01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised"""
        males = ['01', '03', '05', '07', '09', '11', '13', '15', '17', '19', '21', '23']
        females = ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20', '22', '24']

        if gender == "male":
            selected_subjects = np.asarray(males)
            number_subjects = len(males)
            self.subjects_gender = np.full(len(males), "male")
        elif gender == "female":
            selected_subjects = np.asarray(females)
            number_subjects = len(females)
            self.subjects_gender = np.full(len(females), "female")
        else:
            selected_subjects = np.concatenate((males, females), axis=None)
            number_subjects = len(males) + len(females)
            self.subjects_gender = np.concatenate((np.full(len(males), "male"), +np.full(len(females), "female")),
                                                  axis=0)

        classes = {v: k for k, v in self.classes.items()}
        targets = []
        subjects = []

        for audio in os.listdir(path):
            audio_splitted = audio.split(".wav")[0].split('-')
            target = classes[audio_splitted[2]]
            intensity = audio_splitted[3]
            actual_subject = audio_splitted[6]
            if target in emotions and intensity == "02" and actual_subject in selected_subjects:
                audio_path = os.path.join(path, audio)
                [x, Fs] = librosa.load(audio_path, sr=16000)
                self.data.append((x, Fs))
                targets.append(target)
                subject = int(np.where(selected_subjects == actual_subject)[0])
                subjects.append(subject)

        for j in range(0, number_subjects):
            indices = np.where(np.isin(subjects, j))[0].tolist()
            self.subjects.append(indices)

        for original in targets:
            self.targets.append(self.emotions.index(original))

    def get_berlin_dataset(self, path, gender, emotions):
        """ name format = 03a01Fa
        anger=W, boredom=L, disgust=E, anxiety/fear=A, happiness=F, sadness=T, neutral version=N"""
        males = ['03', '10', '11', '12', '15']
        females = ['08', '09', '13', '14', '16']

        if gender == "male":
            selected_subjects = np.asarray(males)
            number_subjects = len(males)
            self.subjects_gender = np.full(len(males), "male")
        elif gender == "female":
            selected_subjects = np.asarray(females)
            number_subjects = len(females)
            self.subjects_gender = np.full(len(females), "female")
        else:
            selected_subjects = np.concatenate((males, females), axis=None)
            number_subjects = len(males) + len(females)
            self.subjects_gender = np.concatenate((np.full(len(males), "male"), +np.full(len(females), "female")),
                                                  axis=0)

        classes = {v: k for k, v in self.classes.items()}
        targets = []
        subjects = []

        for audio in os.listdir(path):
            target = classes[audio[5]]
            actual_subject = audio[0:2]
            if target in emotions and actual_subject in selected_subjects:
                audio_path = os.path.join(path, audio)
                [x, Fs] = librosa.load(audio_path, sr=16000)
                self.data.append((x, Fs))
                targets.append(target)
                subject = int(np.where(selected_subjects == actual_subject)[0])
                subjects.append(subject)

        for j in range(0, number_subjects):
            indices = np.where(np.isin(subjects, j))[0].tolist()
            self.subjects.append(indices)

        for original in targets:
            self.targets.append(self.emotions.index(original))

    def get_enterface05_dataset(self, path, gender, emotions):
        # name format = s1_an_2
        males = ['s1', 's2', 's3', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18',
                 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's27', 's30', 's32', 's34', 's35', 's36', 's37',
                 's38', 's39', 's40', 's41', 's42', 's43']
        females = ['s4', 's5', 's7', 's26', 's28', 's29', 's31', 's33', 's44']

        if gender == "male":
            selected_subjects = np.asarray(males)
            number_subjects = len(males)
            self.subjects_gender = np.full(len(males), "male")
        elif gender == "female":
            selected_subjects = np.asarray(females)
            number_subjects = len(females)
            self.subjects_gender = np.full(len(females), "female")
        else:
            selected_subjects = np.concatenate((males, females), axis=None)
            number_subjects = len(males) + len(females)
            self.subjects_gender = np.concatenate((np.full(len(males), "male"), +np.full(len(females), "female")),
                                                  axis=0)

        classes = {v: k for k, v in self.classes.items()}
        targets = []
        subjects = []

        for audio in os.listdir(path):
            audio_splitted = audio.split(".wav")[0].split('_')
            target = classes[audio_splitted[1]]
            actual_subject = audio_splitted[0]
            if target in emotions and actual_subject in selected_subjects:
                audio_path = os.path.join(path, audio)
                [x, Fs] = librosa.load(audio_path, sr=16000)
                self.data.append((x, Fs))
                targets.append(target)
                subject = int(np.where(selected_subjects == actual_subject)[0])
                subjects.append(subject)

        for j in range(0, number_subjects):
            indices = np.where(np.isin(subjects, j))[0].tolist()
            self.subjects.append(indices)

        for original in targets:
            self.targets.append(self.emotions.index(original))

    def get_cremad_dataset(self, path, gender, emotions):
        # name format = 1001_IEO_FEA_LO
        levels = ['MD', 'HI']

        males = ['1001', '1005', '1011', '1014', '1015', '1016', '1017', '1019', '1022', '1023', '1026', '1027', '1031',
                 '1032', '1033', '1034', '1035', '1036', '1038', '1039', '1040', '1041', '1042', '1044', '1045', '1048',
                 '1050', '1051', '1057', '1059', '1062', '1064', '1065', '1066', '1067', '1068', '1069', '1070', '1071',
                 '1077', '1080', '1081', '1083', '1085', '1086', '1087', '1088', '1090'
                 ]
        females = ['1002', '1003', '1004', '1006', '1007', '1008', '1009', '1010', '1012', '1013', '1018', '1020',
                   '1021', '1024', '1025', '1028', '1029', '1030', '1037', '1043', '1046', '1047', '1049', '1052',
                   '1053',
                   '1054', '1055', '1056', '1058', '1060', '1061', '1063', '1072', '1073', '1074', '1075', '1076',
                   '1078',
                   '1079', '1082', '1084', '1089', '1091'
                   ]

        if gender == "male":
            selected_subjects = np.asarray(males)
            number_subjects = len(males)
            self.subjects_gender = np.full(len(males), "male")
        elif gender == "female":
            selected_subjects = np.asarray(females)
            number_subjects = len(females)
            self.subjects_gender = np.full(len(females), "female")
        else:
            selected_subjects = np.concatenate((males, females), axis=None)
            number_subjects = len(males) + len(females)
            self.subjects_gender = np.concatenate((np.full(len(males), "male"), +np.full(len(females), "female")),
                                                  axis=0)

        classes = {v: k for k, v in self.classes.items()}
        targets = []
        subjects = []

        for audio in os.listdir(path):
            audio_splitted = audio.split(".wav")[0].split('_')
            target = classes[audio_splitted[2]]
            level = audio_splitted[3]
            actual_subject = audio_splitted[0]
            if target in emotions and level in levels and actual_subject in selected_subjects:
                audio_path = os.path.join(path, audio)
                [x, Fs] = librosa.load(audio_path, sr=16000)
                self.data.append((x, Fs))
                targets.append(target)
                subject = int(np.where(selected_subjects == actual_subject)[0])
                subjects.append(subject)

        for j in range(0, number_subjects):
            indices = np.where(np.isin(subjects, j))[0].tolist()
            self.subjects.append(indices)

        for original in targets:
            self.targets.append(self.emotions.index(original))
