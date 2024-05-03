import numpy as np
import time
import cv2 as cv

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def tell_time(start_time, sentence = "Time at:"):

    """
    Function that tells the time since a given start_time
    """
    time_delta = time.time() - start_time
    print(sentence,time_delta/60)

def billiniar_blur_img(img_path, img_size):
    
    """
    Function that applies a billiniar_blur mask 
    to the outside of a circle centered in the image
    """

    # Read image
    img = cv.imread(img_path)
    img = cv.resize(img, (img_size[1], img_size[0]))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    circle_radius = 56 # half of 112 (img.shape[0])

    # Mask
    mask = np.zeros_like(img)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    mask = cv.circle(mask, center, circle_radius, (1, 1, 1), thickness=-1)
    
    # Invert
    mask_inv = 1 - mask
    
    # Apply bilateralFilter
    result = img.copy()
    blurred_inner_outer_img = cv.bilateralFilter(result, 10, 100, 100)
    result[mask_inv == 1] = blurred_inner_outer_img[mask_inv == 1]
    
    return result


def show_blur(img_path, img_size):

    """
    Function that plots original image and the blured one
    """

    img = cv.imread(img_path)
    img = cv.resize(img, (img_size[1], img_size[0]))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # BGR defualt of CV2 and I need it to be rgb

    billiniar = billiniar_blur_img(img_path, img_size)

    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(billiniar), plt.title('Billiniar Blur')
    plt.xticks([]), plt.yticks([])



def segmentation_LAB(img_path, img_size):

    """
    DEPRICATED CODE

    Function that selects pixels based on a threshold 
    and then segments them  

    Depricated in favor of DullRazor because the images
    resulting were not good

    """

    # Read image
    img = cv.imread(img_path)
    img = cv.resize(img, (img_size[1], img_size[0]))
    lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    # Threshold
    skin_lower = np.array([0, 20, 100], dtype=np.uint8)
    skin_upper = np.array([255, 150, 255], dtype=np.uint8)

    dark_lower = np.array([0, 0, 0], dtype=np.uint8)
    dark_upper = np.array([50, 50, 50], dtype=np.uint8)

    # Masks
    skin_mask = cv.inRange(lab_img, skin_lower, skin_upper)
    dark_mask = cv.inRange(lab_img, dark_lower, dark_upper)

    combined_mask = cv.bitwise_or(skin_mask, dark_mask)

    combined_mask_inv = cv.bitwise_not(combined_mask)

    # Result
    result_img = cv.bitwise_and(img, img, mask=combined_mask_inv)

    kernel = np.ones((1, 1), np.uint8)
    result_img = cv.morphologyEx(result_img, cv.MORPH_CLOSE, kernel)

    blurred_img = cv.GaussianBlur(img, (15, 15), 0)
    img_with_blur = np.copy(img)
    img_with_blur[combined_mask > 0] = blurred_img[combined_mask > 0]

    return cv.cvtColor(img_with_blur, cv.COLOR_BGR2RGB)



def dullRazor(img_path, img_size):
    
    """
    Hair Removel Funtion based from : https://github.com/BlueDokk/Dullrazor-algorithm/blob/main/dullrazor.py

    This function can reliably remove darker hairs but
    struggles with brown/lighter hairs
    """

    img=cv.imread(img_path,cv.IMREAD_COLOR)
    
    #Gray scale
    grayScale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    #Black hat filter
    kernel = cv.getStructuringElement(1,(9,9)) 
    blackhat = cv.morphologyEx(grayScale, cv.MORPH_BLACKHAT, kernel)
    
    #Gaussian filter
    bhg= cv.GaussianBlur(blackhat,(3,3),cv.BORDER_DEFAULT)
    
    #Binary thresholding (MASK)
    ret,mask = cv.threshold(bhg,10,255,cv.THRESH_BINARY)
    
    #Replace pixels of the mask
    dst = cv.inpaint(img,mask,6,cv.INPAINT_TELEA)

    img = cv.resize(img, (img_size[1], img_size[0]))
    mask = cv.resize(mask, (img_size[1], img_size[0]))
    dst = cv.resize(dst, (img_size[1], img_size[0]))

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)

    return img, mask, dst


def show_dullRazor(img_path, img_size):

    """
    Function that plots the original image, the removed mask 
    and the imaged without the hair
    """

    img, mask, dst = dullRazor(img_path, img_size)
    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(mask), plt.title('Mask')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(dst), plt.title('Clean')
    plt.xticks([]), plt.yticks([])


def circle_mask(img_path, img_size):

    """
    Function that creates a mask of dark pixels outside of a centred circle
    based on: https://stackoverflow.com/a/67640459

    """

    img = cv.imread(img_path)
    img = cv.resize(img, (img_size[1], img_size[0]))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # define circle
    center = (img.shape[1] // 2, img.shape[0] // 2)
    radius = 70

    # create mask
    mask = np.zeros(img_size, dtype=np.uint8)
    mask = cv.circle(mask, center, radius, 255, -1)
    color = np.full_like(img, (0,0,0))

    # apply mask
    masked_img = cv.bitwise_and(img, img, mask=mask)
    masked_color = cv.bitwise_and(color, color, mask=255-mask)
    result = cv.add(masked_img, masked_color)

    return img, result

def show_circle_mask(img_path, img_size):

    """
    Function that plots the original image
    and the imaged with the dark mask
    """

    img, mask = circle_mask(img_path, img_size)
    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(mask), plt.title('Mask')
    plt.xticks([]), plt.yticks([])

