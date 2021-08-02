from utils import save_read_classifier as srcl
from utils import read_save_images as rs
from utils import folder_operations as fo
from sklearn import metrics
import time
from utils import refactor_image as rf
from utils import matrix
import os

def main():
    # gpu 1 without torch, gpu 0 with torch (using nvidia-smi in terminal the gpus are inverted)
    selectGpu = 0
    # selectGpu = 1
    if selectGpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    isUpperGland = True
    if isUpperGland:
        clahe_images, masks, filenames = fo.read_Masks_CLAHE_Up()
    else:
        clahe_images, masks, filenames = fo.read_Masks_CLAHE_Low()

    k = 0
    name = '10 CLAHE'
    for filename in filenames:
        if filename == name:
            print("Name:", filename, "-", "index:", k)
            break
        k = k + 1

    if isUpperGland:
        clf = srcl.loadSVM("classifier/classifier_All_10x10_coordinate_Up_pca")
    else:
        clf = srcl.loadSVM("classifier/classifier_All_10x10_coordinate_Low_pca")

    # matrix of image that will be refactor
    test_cov, test_labels = matrix.read_matrix('Matrices/Single/matrix_svm_10_10x10_coordinate')
    test_labels = test_labels[0]

    time_start_predict = time.time()
    predict = clf.predict(test_cov)
    f1score = metrics.f1_score(test_labels, predict)
    accuracy = metrics.accuracy_score(test_labels, predict)
    time_end_predict = time.time()

    print(time_end_predict - time_start_predict)

    count, count1, count2, count3, count4 = 0, 0, 0, 0, 0
    for i in range(len(predict)):
        if test_labels[i] == predict[i]:
            count += 1
        if predict[i] == 1 and test_labels[i] == 1:
            count1 += 1
        elif predict[i] == 1 and test_labels[i] == 0:
            count3 += 1
        if test_labels[i] == 1:
            count2 += 1
        elif test_labels[i] == 0:
            count4 += 1

    print("Blocchi matched dalla predict: ", count)
    print("Blocchi totali: ", len(test_labels))
    print('Accuracy: ', accuracy)
    print('F1_score: ', f1score)
    print("Meibomio selezionato % (% di 1): ", count1/count2 * 100)
    print("Meibomio perso %: ", 100 - (count1/count2 * 100))
    print("Parte non corrispondente al Meibomio selezionata %: ", count3/count4 * 100)

    tile = 10
    rec_img, rec_mask = rf.refactor_img(clahe_images[k], predict, tile)
    rs.saveImage("Result/mask/rec_mask_All_10x10_coordinate of image" + name, rec_mask)
    rs.saveImage("Result/img/rec_img_All_10x10_coordinate of image" + name, rec_img)

if __name__ == "__main__":
    main()
