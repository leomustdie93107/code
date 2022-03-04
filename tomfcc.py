# -*- coding:utf-8 -*-
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
import cv2
directoryName = "C:/Users/justi/OneDrive/桌面/新增資料夾/語音辨識/all_train"
i = -1
for filename in os.listdir(directoryName):
    i = i + 1
    directoryName = "C:/Users/justi/OneDrive/桌面/新增資料夾/語音辨識/all_train"
    resultsDirectory = 'C:/Users/justi/OneDrive/data/'+ str(i)
    directoryName = directoryName +'/'+ os.path.splitext(filename)[0]
    # make a new folder in this directory to save our results in
    if not os.path.exists(resultsDirectory):
        os.makedirs(resultsDirectory)
    # get MFCCs for every .wav file in our specified directory 
    for filename in os.listdir(directoryName):
        print(filename)
        if filename.endswith('.wav'): # only get MFCCs from .wavs
            # read in our file
            frequency_sampling, audio_signal = wav.read(directoryName + "/" +filename)
            print(frequency_sampling,audio_signal)
            audio_signal = audio_signal[:15000]
            features_mfcc = mfcc(audio_signal, frequency_sampling,numcep = 40,winlen=0.03,winstep=0.01)
            #最主要轉的是這段
            features_mfcc = features_mfcc.T
            features_mfcc = ((features_mfcc-features_mfcc.min())/(features_mfcc.max()-features_mfcc.min()))*255
            #features_mfcc = np.uint8(features_mfcc)
            outputFile = resultsDirectory + "/" + os.path.splitext(filename)[0]+".jpg"
            #plt.savefig(outputFile,features_mfcc)
            cv.imwrite(outputFile,features_mfcc)


            
        