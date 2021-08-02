import itertools


def tile(img, d):
    crop_img = []
    w, h = img.shape[0], img.shape[1]

    #scorre verticalmente!
    grid = list(itertools.product(range(0, h - h % d, d), range(0, w - w % d, d)))
    for i, j in grid:
        crop_img.append(img[j:j+d, i:i+d])
    return crop_img


def calculate_labels(crop_mask):
    labels = []
    for i in range(len(crop_mask)):
        counter = 0
        for j in range(len(crop_mask[i])):
            for k in range(len(crop_mask[i][j])):
                if crop_mask[i][j][k] == 255:
                    counter += 1
        if (counter/(crop_mask[i].shape[0] * crop_mask[i].shape[1])) >= 0.7:
            labels.append(1)
        else:
            labels.append(0)
    return labels




