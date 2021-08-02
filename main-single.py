import folder_operations as fo
import numpy as np
import gabor
import split_image
import matrix
import refactor_image as rf
from copy import deepcopy

def main():
    cov_matrix = []
    labels = []
    x = []
    y = []
    isUpperGland = False
    create_matrix = False

    if isUpperGland:
        clahe_images, masks, filenames = fo.read_Masks_CLAHE_Up()
    else:
        clahe_images, masks, filenames = fo.read_Masks_CLAHE_Low()

    # select the image that you want create the matrix
    start = 0
    step = 1
    stop = start + 1
    tile = 10
    for i in range(start, stop, step):
        print(filenames[i])
        crop_mask = split_image.tile(masks[i], tile)
        label = split_image.calculate_labels(crop_mask)
        num_theta = 2
        gabor_ker_size = 129
        lam = np.arange(1.010, 1.016, 0.005)
        Gr, Gi = gabor.get_gabor_kernel_bank(gabor_ker_size, num_theta, lam)
        features = np.array(gabor.apply_gabor(clahe_images[i], Gr, Gi))
        box_features = []
        for j in range(features.shape[0]):
            img_feature = rf.reconstruct_feature(848, 1280, features[j])
            crop_feature = split_image.tile(img_feature, tile)
            if (i - start) == 0 and j == 0:
                x, y = rf.extract_coordinate(clahe_images[0].shape[0], len(crop_feature), tile)
                x = [x[z]/max(x) for z in range(0, len(x))]
                y = [y[z]/max(y) for z in range(0, len(y))]
            if j == 0:
                x_array = deepcopy(x)
                y_array = deepcopy(y)
            box_features.append(crop_feature)
        labels.append(label)
        for k in range(0, len(box_features[0])):
            cube = []
            for j in range(0, len(box_features)):
                cube.append(box_features[j][k])
            cube = np.array(cube).reshape((num_theta * 2, tile * tile))
            processed_features = np.dot(cube, np.transpose(cube))
            processed_features = np.reshape(processed_features, -1)
            processed_features = np.append(processed_features, x_array[k])
            processed_features = np.append(processed_features, y_array[k])
            cov_matrix.append(processed_features)

        labels = [item for sublist in labels for item in sublist]

        name = filenames[i].replace(' CLAHE', '')
        if create_matrix:
            if isUpperGland:
                name = "Matrices/Single/matrix_svm_" + name
                                     + "_" + str(tile) + "x"
                                     + str(tile) + "_coordinate_up"
                matrix.create_matrix(cov_matrix, labels, name)
            else:
                name = "Matrices/Single/matrix_svm_" + name
                                     + "_" + str(tile) + "x"
                                     + str(tile) + "_coordinate_low"
                matrix.create_matrix(cov_matrix, labels, name)
    print('fine')



if __name__ == "__main__":
    main()
