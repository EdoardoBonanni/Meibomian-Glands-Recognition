import numpy as np
from copy import deepcopy

def refactor_img(img, labels, dimKernel):
    cnt_row, cnt_col = 0, 0
    rec_mask = np.zeros((img.shape[0], img.shape[1]))
    rec_img = deepcopy(img)
    for i in range(len(labels)):
        if (cnt_row * dimKernel + dimKernel - 1) < img.shape[0]:
            for j in range(dimKernel):
                for k in range(dimKernel):
                    if labels[i] == 0:
                        rec_mask[k + (cnt_row * dimKernel)][j + (cnt_col * dimKernel)] = 0
                        rec_img[k + (cnt_row * dimKernel)][j + (cnt_col * dimKernel)] = 0
                    else:
                        rec_mask[k + (cnt_row * dimKernel)][j + (cnt_col * dimKernel)] = 255
            cnt_row += 1

        else:
            cnt_row = 0
            cnt_col += 1
            for j in range(dimKernel):
                for k in range(dimKernel):
                    if labels[i] == 0:
                        rec_mask[k + (cnt_row * dimKernel)][j + (cnt_col * dimKernel)] = 0
                        rec_img[k + (cnt_row * dimKernel)][j + (cnt_col * dimKernel)] = 0
                    else:
                        rec_mask[k + (cnt_row * dimKernel)][j + (cnt_col * dimKernel)] = 255

            cnt_row += 1
    #rs.visualizeImage(rec_mask)
    #rs.visualizeImage(rec_img)
    return rec_img, rec_mask

def reconstruct_feature(shape0, shape1, feature):
    k = 0
    img_feature = np.zeros((shape0, shape1))
    for i in range(shape0):
        for j in range(shape1):
            img_feature[i][j] = feature[k]
            k += 1
    return img_feature

def extract_coordinate(shape0, length, dimKernel):
    cnt_row, cnt_col = 0, 0
    x = []
    y = []
    for i in range(length):
        if (cnt_row * dimKernel + dimKernel - 1) < shape0:
            x_value = []
            y_value = []
            for j in range(dimKernel):
                for k in range(dimKernel):
                    x_value.append(k + (cnt_row * dimKernel) )
                    y_value.append(j + (cnt_col * dimKernel))
            x.append(np.mean(x_value))
            y.append(np.mean(y_value))
            cnt_row += 1

        else:
            cnt_row = 0
            cnt_col += 1
            x_value = []
            y_value = []
            for j in range(dimKernel):
                for k in range(dimKernel):
                    x_value.append(k + (cnt_row * dimKernel))
                    y_value.append(j + (cnt_col * dimKernel))
            x.append(np.mean(x_value))
            y.append(np.mean(y_value))
            cnt_row += 1
    return x, y