from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import numpy as np
import cv2
from copy import deepcopy

def createSuperpixelWithMask(img):
    # load the image and apply SLIC and extract (approximately)
    # the supplied number of segments
    image = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    segments = slic(image, n_segments=180, sigma=5, start_label=1)

    # show the output of SLIC
    segmentedImage = mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments)
    return segmentedImage, segments

def selectPoint(img, segments):

    image = deepcopy(img)

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            segment = segments[y, x]
            if(img[y, x] == np.array([0, 1, 0])).all():
                img[segments == segment] = image[segments == segment]
            else:
                #BGR color
                img[segments == segment] = np.array([0, 1, 0])
            cv2.imshow("Click any key to close", img)

    cv2.namedWindow("Click any key to close")
    cv2.setMouseCallback("Click any key to close", on_EVENT_LBUTTONDOWN)
    while(True):
        cv2.imshow("Click any key to close", img)
        if cv2.waitKey(0):
            break
    cv2.destroyAllWindows()
    mask = np.zeros(img.shape[:2], dtype="uint8")
    greenElement = np.array([0, 1, 0])
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            res = (img[i, j] == greenElement)
            if res[0] and res[1] and res[2]:
                mask[i, j] = 255
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mask



