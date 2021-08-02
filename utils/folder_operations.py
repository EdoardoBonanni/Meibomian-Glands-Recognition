import os
import cv2
from utils import matrix

def read_TIFF(file):
    image = cv2.imread(file, -1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def create_CLAHE():
    imagesTIFF = []
    imagesCLAHE = []
    filenames = []
    clahe8x8 = cv2.createCLAHE(clipLimit=40, tileGridSize=(8, 8))
    with os.scandir('TIFF images/') as entries:
        for entry in entries:
            if ('.TIFF' in entry.name):
                entryname = entry.name
                entryname = entryname.replace('.TIFF', '')
                image = read_TIFF('TIFF images/' + entry.name)
                cl = clahe8x8.apply(image)
                imagesTIFF.append(image)
                imagesCLAHE.append(cl)
                filenames.append('Clahe images/' + entryname + ' CLAHE')
    return imagesTIFF, imagesCLAHE, filenames

def read_CLAHE_Up():
    imagesCLAHE = []
    filenames = []
    with os.scandir('Clahe images/Upper Gland/') as entries:
        for entry in entries:
            if ('.jpg' in entry.name):
                name = entry.name.replace('.jpg', '')
                img = cv2.imread('Clahe images/Upper Gland/' + name + '.jpg', 0)
                imagesCLAHE.append(img)
                filenames.append(name)
    return imagesCLAHE, filenames

def read_CLAHE_Low():
    imagesCLAHE = []
    filenames = []
    with os.scandir('Clahe images/Lower Gland/') as entries:
        for entry in entries:
            if ('.jpg' in entry.name):
                name = entry.name.replace('.jpg', '')
                img = cv2.imread('Clahe images/Lower Gland/' + name + '.jpg', 0)
                imagesCLAHE.append(img)
                filenames.append(name)
    return imagesCLAHE, filenames

def read_Masks_CLAHE_Up():
    imagesCLAHE = []
    masks = []
    filenames = []
    with os.scandir('Masks/Upper Gland/') as entries:
        for entry in entries:
            if ('.mat' in entry.name):
                name = entry.name.replace('.mat', '')
                mask = matrix.read_matrix_mask('Masks/Upper Gland/' + name)
                name = name.replace('mask_', '')
                img = cv2.imread('Clahe images/Upper Gland/' + name + '.jpg', 0)
                imagesCLAHE.append(img)
                masks.append(mask)
                filenames.append(name)
    return imagesCLAHE, masks, filenames

def read_Masks_CLAHE_Low():
    imagesCLAHE = []
    masks = []
    filenames = []
    with os.scandir('Masks/Lower Gland/') as entries:
        for entry in entries:
            if ('.mat' in entry.name):
                name = entry.name.replace('.mat', '')
                mask = matrix.read_matrix_mask('Masks/Lower Gland/' + name)
                name = name.replace('mask_', '')
                img = cv2.imread('Clahe images/Lower Gland/' + name + '.jpg', 0)
                imagesCLAHE.append(img)
                masks.append(mask)
                filenames.append(name)
    return imagesCLAHE, masks, filenames

def save_images(imagesCLAHE, filenames):
    for i in range(len(filenames)):
        cv2.imwrite(filenames[i] + '.jpg', imagesCLAHE[i])

def create_masks_images_Up(masks, filenames):
    for i in range(0, len(masks)):
        cv2.imwrite("Masks images/Upper Gland/mask_" + filenames[i] + '.jpg', masks[i])
        print('')

def create_masks_images_Low(masks, filenames):
    for i in range(0, len(masks)):
        cv2.imwrite("Masks images/Lower Gland/mask_" + filenames[i] + '.jpg', masks[i])
        print('')
