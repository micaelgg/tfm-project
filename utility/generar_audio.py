#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 20:18:42 2018

@author: mike

Funciones navaja suiza
"""
#import moviepy.editor as mp
import numpy as np
import os
from pydub import AudioSegment


def avi_to_wav():
    numero_subject = np.arange(1, 45)
    numero_subject = np.delete(numero_subject, 5)  # eliminar el subjeto 6
    emociones = ("anger", "disgust", "fear", "happiness", "sadness", "surprise")
    numero_sentence = np.arange(1, 6)
    path_base = "/home/mike/Documents/TFM/programacion/databases/eNTERFACE05/enterface database"

    path_audios = "/home/mike/Documents/TFM/programacion/databases/eNTERFACE05/audios/stereo/"
    for subjeto in numero_subject:
        for emocion in emociones:
            for sentencia in numero_sentence:
                subjeto = str(subjeto)
                sentencia = str(sentencia)
                directorio = path_base + "/subject " + subjeto + "/" + emocion + "/sentence " + sentencia + "/"
                nombre_archivo = "s" + subjeto + "_" + emocion[0:2] + "_" + sentencia
                clip = mp.VideoFileClip(directorio + nombre_archivo + ".avi")
                clip.audio.write_audiofile(directorio + nombre_archivo + ".wav")
                clip.audio.write_audiofile(path_audios + emocion + "/" + nombre_archivo + ".wav")
                clip.audio.write_audiofile(path_audios + "all/" + nombre_archivo + ".wav")


def corregir_nombres():
    numero_subject = np.arange(1, 46)
    emociones = ("anger", "disgust", "fear", "happiness", "sadness", "surprise")
    numero_sentence = np.arange(1, 6)
    path_base = "/home/mike/Documents/TFM/programacion/databases/eNTERFACE05/enterface database"
    for subjeto in numero_subject:
        for emocion in emociones:
            for sentencia in numero_sentence:
                subjeto = str(subjeto)
                sentencia = str(sentencia)
                directorio = path_base + "/subject " + subjeto + "/" + emocion + "/sentence " + sentencia + "/"
                nombre_archivo = "s" + subjeto + "_" + emocion[0:2] + "_" + sentencia
                nombre_erroneo = "s_" + subjeto + "_" + emocion[0:2] + "_" + sentencia
                if (os.path.isfile(directorio + nombre_erroneo + ".avi")):
                    os.rename(directorio + nombre_erroneo + ".avi", directorio + nombre_archivo + ".avi")
                    print
                    "corregido " + nombre_erroneo


def stero_to_mono():
    path_audios = "/home/mike/Documents/TFM/programacion/databases/eNTERFACE05/audios/stereo/"
    path_out = "/home/mike/Documents/TFM/programacion/databases/eNTERFACE05/audios/mono/"
    emociones = ("anger", "disgust", "fear", "happiness", "sadness", "surprise")
    for emocion in emociones:
        path_to_wav = path_audios + emocion
        path_out_wav = path_out + emocion
        files = os.listdir(path_to_wav)
        for f in files:
            sound = AudioSegment.from_wav(path_to_wav + "/" + f)
            sound = sound.set_channels(1)
            sound.export(path_out_wav + "/" + f, format="wav")


def stero_to_mono_misma_carpeta():
    path_audios = "/home/mike/TFM/programacion/datasets/RAVDESS/all-mono-wav-audios"
    path_out = "/home/mike/TFM/programacion/datasets/RAVDESS/all-mono-wav-audios"
    emociones = ("anger", "disgust", "fear", "happiness", "sadness", "surprise")
    files = os.listdir(path_audios)
    for f in files:
        sound = AudioSegment.from_wav(path_audios + "/" + f)
        sound = sound.set_channels(1)
        sound.export(path_audios + "/" + f, format="wav")


stero_to_mono_misma_carpeta()
