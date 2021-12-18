import os
from os.path import isfile, join
import re

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


def iterative_imgs(clue_tier):
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
            # print(crop)
            # crop = cv2.GaussianBlur(crop, (3,3), cv2.BORDER_DEFAULT)
            icons.append(crop)
            twidth += 40
            bwidth += 40
            # print(twidth)
            # print(bwidth)
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
            cv2.imwrite(destination + "icon_" + str(index) + ".png", icon_list[i][j])
            index += 1


def icon_cutter(image):
    # Cuts the icons out of the original photo
    theight = 52   # was 53
    twidth = 25
    bheight = 84   # was 83
    bwidth = 58
    icons = []
    for i in range(9):
        temp = image
        crop = temp[theight:bheight, twidth:bwidth]
        icons.append(crop)
        twidth += 40
        bwidth += 40
    return icons


def image_processor(mypath, file_names):
    # Adds background and Gauss to folder images
    blurred_imgs = []
    for i in range(len(file_names)):
        if file_names[i] == "Blank_image.png":
            image = cv2.imread(mypath + file_names[i], cv2.IMREAD_UNCHANGED)
            blurred_imgs.append(image)
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.show()
        else:
            image = cv2.imread(mypath + file_names[i], cv2.IMREAD_UNCHANGED)
            trans_mask = image[:, :, 3] == 0
            image[trans_mask] = [41, 31, 10, 255]
            image2 = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image3 = cv2.GaussianBlur(image2, (3, 5), cv2.BORDER_DEFAULT)
            blurred_imgs.append(image3)
            # plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
            # plt.show()

    return blurred_imgs


def histo_runner(mypath, file_names, mypathIcon, file_names_icons):

    blurred_screenshot_images = []
    for i in range(len(file_names_icons)):
        image = cv2.imread(mypathIcon + file_names_icons[i], cv2.IMREAD_UNCHANGED)
        crop = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
        blurred_screenshot_images.append(crop)

    # A few notes
    # - The font used on the total value box is Trajan 3 Regular
    # - The font used on item quantities is Runescape UF font

    file_images = []
    print("getting the images")
    for i in range(len(file_names)):
        image = cv2.imread(mypath + file_names[i])
        file_images.append(image)

    blurred_file_images = image_processor(mypath, file_names)

    complete_list = []

    # Iterative Compare Hist
    print("Comparing hist...")
    for i in range(len(blurred_screenshot_images)):
        item_name = ''
        currentBest = 0.0
        indexOfBest = 0
        for j in range(len(file_names)):
            print("Comparing the "+str(i+1)+" icon with " + file_names[j])
            hsv_test1 = cv2.cvtColor(blurred_screenshot_images[i], cv2.COLOR_BGR2HSV)
            hsv_test2 = cv2.cvtColor(blurred_file_images[j], cv2.COLOR_BGR2HSV)
            h_bins = 50
            s_bins = 60
            histSize = [h_bins, s_bins]
            h_ranges = [0, 180]
            s_ranges = [0, 256]
            ranges = h_ranges + s_ranges
            channels = [0, 1]
            hist_test1 = cv2.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_test2 = cv2.calcHist([hsv_test2], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            base_base = cv2.compareHist(hist_test1, hist_test2, 0)
            if base_base > currentBest:
                currentBest = base_base
                indexOfBest = j
            #print("Distance between the first icon and " +file_names[j]+ " is " +str(base_base))
            # print("Current best is " + file_names[indexOfBest])
        print("Item " +file_names_icons[i]+ " discovered was " + file_names[indexOfBest] + " with a score of " + str(currentBest))


def last_4chars(x):
    return(x[-4:])


clue_tier = -1
while clue_tier < 0 or clue_tier > 1:
    print("What reward? (only 0 and 1)")
    clue_tier = int(input())
tier = ''
mypath = 'images/Reward Files/'
if clue_tier == 0:
    mypath += 'Easy rewards/'
    tier = 'easy'
elif clue_tier == 1:
    mypath += 'Medium rewards/'
    tier = 'medium'
elif clue_tier == 2:
    mypath += 'Hard rewards/'
    tier = 'hard'
elif clue_tier == 3:
    mypath += 'Elite rewards/'
    tier = 'elite'
else:
    mypath += 'Master rewards/'
    tier = 'master'

mypathIcon = 'images/cropped_icons/new_icons/'
if clue_tier == 0:
    mypathIcon += 'Easy Icons/'
    tier = 'easy'
elif clue_tier == 1:
    mypathIcon += 'Medium Icons/'
    tier = 'medium'
elif clue_tier == 2:
    mypathIcon += 'Hard Icons/'
    tier = 'hard'
elif clue_tier == 3:
    mypathIcon += 'Elite Icons/'
    tier = 'elite'
else:
    mypathIcon += 'Master Icons/'
    tier = 'master'

choice = -1
while choice < 0 or choice > 1:
    print("what are you doing this time?")
    choice = int(input())

file_names = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
file_names_icons = [f for f in os.listdir(mypathIcon) if isfile(join(mypathIcon, f))]
file_names_icons.sort(key=lambda f: int(re.sub('\D', '', f)))

print(file_names)
print(file_names_icons)

if choice == 0:
    iterative_imgs(clue_tier)
else:
    histo_runner(mypath, file_names, mypathIcon, file_names_icons)
