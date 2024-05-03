import pandas as pd
import cv2 as cv
import os
from utils import dullRazor, circle_mask, billiniar_blur_img

# PATHS
paths_train = os.listdir( "./train")
paths_test = os.listdir( "./test")
img_size = (112,150)

# Folder creation
dullRazor_folder = "./new_data/dullRazor_images/"
os.makedirs(dullRazor_folder, exist_ok=True)
paths_dullRazor = os.listdir("./train")

circle_mask_folder = "./new_data/circle_mask_images/"
os.makedirs(circle_mask_folder, exist_ok=True)

blur_folder = "./new_data/blurred_images/"
os.makedirs(blur_folder, exist_ok=True)

test_folder = "./new_data/test_folder/"
os.makedirs(test_folder, exist_ok=True)

#Control image gen
use_dullRazor = False
use_circle_mask = False
use_biliniar_blur = False
use_test_folder = False

#Image gen
if use_dullRazor:
    for image_path in paths_train:
        _, _,img = dullRazor(f"./train/{image_path}", img_size)         
        filename = os.path.splitext(os.path.basename(image_path))[0]
        cv.imwrite(os.path.join(dullRazor_folder, f"{filename}.jpg"), cv.cvtColor(img, cv.COLOR_BGR2RGB))
    print("Done DullRazor") 

if use_circle_mask:
    for image_path in paths_dullRazor:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        _, img = circle_mask(os.path.join(dullRazor_folder, f"{filename}.jpg"), img_size)
        cv.imwrite(os.path.join(circle_mask_folder, f"{filename}.jpg"), cv.cvtColor(img, cv.COLOR_BGR2RGB))
    print("Done Circle Mask")

if use_biliniar_blur:
    for image_path in paths_dullRazor:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        img = billiniar_blur_img(os.path.join(dullRazor_folder, f"{filename}.jpg"), img_size)
        cv.imwrite(os.path.join(blur_folder, f"{filename}.jpg"), cv.cvtColor(img, cv.COLOR_BGR2RGB)) 
    print("Done Blur")

if use_test_folder:
    for image_path in paths_test:
        _, _,img = dullRazor(f"./test/{image_path}", img_size)         
        filename = os.path.splitext(os.path.basename(image_path))[0]
        cv.imwrite(os.path.join(test_folder, f"{filename}.jpg"), cv.cvtColor(img, cv.COLOR_BGR2RGB))
    print("Done Test DullRazor") 

print("Done") 