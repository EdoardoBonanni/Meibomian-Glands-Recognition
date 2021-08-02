import cv2
import folder_operations as fo

def readImages():
    imagesTIFF, imagesCLAHE, filenames = fo.create_CLAHE()
    return imagesTIFF, imagesCLAHE, filenames

def saveClahe(imagesCLAHE, filenames):
    for i in range(0, len(imagesCLAHE)):
        cv2.imwrite(filenames[i] + '.jpg', imagesCLAHE[i])

def visualizeImages(images):
    for i in range(0, len(images)):
        cv2.imshow('images' + str(i), images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualizeImage(image):
  cv2.imshow('image', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def saveImage(filename, img):
  cv2.imwrite(filename + ".jpg", img)
