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


def icon_cutter(image):
    # Cuts the icons out of the original photo
    theight = 53
    twidth = 25
    bheight = 83
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
            image3 = cv2.GaussianBlur(image2, (1, 3), cv2.BORDER_DEFAULT)
            blurred_imgs.append(image3)
            # plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
            # plt.show()

    return blurred_imgs


def main():
    clue_tier = 1
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

    img = cv2.imread("images/Reward Files/sample clues/clue1.png")

    icons = icon_cutter(img)

    blurred_icons = []
    for i in range(9):
        crop = cv2.GaussianBlur(icons[i], (1, 3), cv2.BORDER_DEFAULT)
        blurred_icons.append(crop)

    # A few notes
    # - The font used on the total value box is Trajan 3 Regular
    # - The font used on item quantities is Runescape UF font

    file_names = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    print(file_names)

    imgs = []
    print("getting the images")
    for i in range(len(file_names)):
        image = cv2.imread(mypath + file_names[i])
        imgs.append(image)

    blurred_imgs = image_processor(mypath, file_names)

    complete_list = []

    '''test1 = blurred_imgs[0]
    test2 = blurred_icons[0]

    hsv_test1 = cv2.cvtColor(test1, cv2.COLOR_BGR2HSV)
    hsv_test2 = cv2.cvtColor(test2, cv2.COLOR_BGR2HSV)
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
    print(base_base)'''

    for i in range(9):
        item_name = ''
        currentBest = 0.0
        indexOfBest = 0
        for j in range(len(file_names)):
            # print("Comparing the "+str(i+1)+" icon with " + file_names[j])
            hsv_test1 = cv2.cvtColor(icons[i], cv2.COLOR_BGR2HSV)
            hsv_test2 = cv2.cvtColor(imgs[j], cv2.COLOR_BGR2HSV)
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
            # print("Distance between the first icon and " +file_names[j]+ " is " +str(base_base))
            # print("Current best is " + file_names[indexOfBest])
        print("Item discovered was " + file_names[indexOfBest] + " with a score of " + str(currentBest))


        # todo: implement image comparisons


if __name__ == '__main__':
    main()

# This only existed to pull images from a folder to test cropping
# Ignore this as it currently has no use
'''def iterative_imgs():
    imgs = []
    print("getting the images")
    for i in range(10):
        img = cv2.imread("images/Reward Files/sample clues/clue" + str(i + 1) + ".png")
        imgs.append(img)

    icon_list = []

    print("Cropping and making icons")
    for i in range(len(imgs)):
        theight = 53
        twidth = 25
        bheight = 83
        bwidth = 58
        icons = []
        for j in range(9):
            temp = imgs[i]
            crop = temp[theight:bheight, twidth:bwidth]
            print(crop)
            crop = cv2.GaussianBlur(crop, (3,3), cv2.BORDER_DEFAULT)
            icons.append(crop)
            twidth += 40
            bwidth += 40
            print(twidth)
            print(bwidth)
        icon_list.append(icons)

    destination = 'images/cropped_icons'
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
            cv2.imwrite(destination + "\icon " + str(index) + ".png", icon_list[i][j])
            index += 1
    
    for i in range(len(files)):
        image = cv2.imread(mypath + files[i], cv2.IMREAD_UNCHANGED)
        trans_mask = image[:,:,3] == 0
        image[trans_mask] = [41, 31, 10, 255]
        new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(mypath + files[i], new_img)'''
