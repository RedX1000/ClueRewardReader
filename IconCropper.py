import os
from os.path import isfile, join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from skimage.metrics import structural_similarity
from tensorflow import keras
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path


def iterative_imgs(mypath, file_names):

    imgs = []
    print("getting the images")
    for i in range(len(file_names)):
        img = cv2.imread(mypath + file_names[i])
        imgs.append(img)
        print(imgs[i])

    icon_list = []

    print("Cropping and making icons")
    for i in range(len(imgs)):
        theight = 52  # Was 53
        twidth = 25
        bheight = 84  # Was 83
        bwidth = 59 #  Was 58
        icons = []
        for j in range(9):
            temp = imgs[i]
            crop = temp[theight:bheight, twidth:bwidth]
            #print(crop)
            #crop = cv2.GaussianBlur(crop, (3,3), cv2.BORDER_DEFAULT)
            icons.append(crop)
            twidth += 40
            bwidth += 40
            #print(twidth)
            #print(bwidth)
        icon_list.append(icons)

    destination = 'images/cropped_icons/new_icons/'
    exists = os.path.exists(destination)
    if not exists:
        os.makedirs(destination)
        print("Destination created")

    index = 0

    for i in range(len(icon_list)):
        for j in range(len(icon_list[i])):
            print(str(i) + " " + str(j))
            print(icon_list[i][j])

    for i in range(len(icon_list)):
        for j in range(len(icon_list[i])):
            print(index)
            cv2.imwrite(destination + "/icon " + str(index) + ".png", icon_list[i][j])
            index += 1

clue_tier = 0
tier = ''
mypath = 'images/Reward Files/sample clues/'
if clue_tier == 0:
    mypath += 'Easys/'
    tier = 'easy'
elif clue_tier == 1:
    mypath += 'Mediums/'
    tier = 'medium'
elif clue_tier == 2:
    mypath += 'Hards/'
    tier = 'hard'
elif clue_tier == 3:
    mypath += 'Elites/'
    tier = 'elite'
else:
    mypath += 'Masters/'
    tier = 'master'

file_names = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
print(file_names)

iterative_imgs(mypath, file_names)