from utils import folder_operations as fo
import numpy as np
from utils import gabor
from utils import split_image
from matplotlib import pyplot as plt
from utils import refactor_image as rf
from utils import matrix
from utils import balance_dataset as bd
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

    start = 0
    step = 1
    stop = len(clahe_images)
    tile = 10
    balance_dataset = 30
    for i in range(start, stop, step):
        print(filenames[i], i, stop)
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
            # plt.imshow(features[j,:].reshape(img_feature.shape[0], img_feature.shape[1]), cmap='jet')
            # plt.title('min %f max %f'%(np.min(features[j,:]), np.max(features[j,:])))
            #plt.axis('off')
            plt.show()
            crop_feature = split_image.tile(img_feature, tile)
            if i-start == 0 and j == 0:
                x, y = rf.extract_coordinate(clahe_images[0].shape[0], len(crop_feature), tile)
                x = [x[z]/max(x) for z in range(0, len(x))]
                y = [y[z]/max(y) for z in range(0, len(y))]
            if j == 0:
                x_array = deepcopy(x)
                y_array = deepcopy(y)
            box_features.append(crop_feature)
        if stop - start > 1:
            box_features, label, x_array, y_array = bd.balance_data_coordinate(box_features, label, balance_dataset, x_array, y_array)
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

    if create_matrix:
        if isUpperGland:
            name_file = "Matrices/matrix_svm_" + str(int((stop - start)/step)) + "Elements_" + str(tile) + "x" + str(tile) + "_coordinate_Up"
            matrix.create_matrix(cov_matrix, labels, name_file)
        else:
            name_file = "Matrices/matrix_svm_" + str(int((stop - start) / step)) + "Elements_" + str(tile) + "x" + str(tile) + "_coordinate_Low"
            matrix.create_matrix(cov_matrix, labels, name_file)
    print('fine')

if __name__ == "__main__":
    main()
