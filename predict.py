from utils import save_read_classifier as srcl
from sklearn import metrics
import time
from utils import matrix
import thundersvm
import os
import warnings

def main():
    warnings.filterwarnings('ignore')

    # gpu 1 without torch, gpu 0 with torch (using nvidia-smi in terminal the gpus are inverted)
    selectGpu = 0
    # selectGpu = 1
    if selectGpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    isUpperGland = True
    # change kernel type if you want
    if isUpperGland:
        clf = srcl.loadSVM("classifier/classifier_All_10x10_coordinate_Up")
        test_cov, test_labels = matrix.read_matrix('Dataset/test_low')
    else:
        clf = srcl.loadSVM("classifier/classifier_All_10x10_coordinate_Low")
        test_cov, test_labels = matrix.read_matrix('Dataset/test_up')

    test_labels = test_labels[0]

    time_start_predict = time.time()
    print(time_start_predict)
    predict = clf.predict(test_cov)
    print('predict done')

    f1score = metrics.f1_score(test_labels, predict)
    accuracy = metrics.accuracy_score(test_labels, predict)
    print(metrics.classification_report(test_labels, predict))
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

    # change log destination file if you use another kernel type
    if isUpperGland:
        f = open("log/all_10x10_coordinate_up.txt", "w")
        f.write("all_10x10_coordinate_up \n")
    else:
        f = open("log/all_10x10_coordinate_low.txt", "w")
        f.write("all_10x10_coordinate_low \n")
    f.write("Blocchi matched dalla predict: " + str(count) + '\n')
    f.write("Blocchi totali: " + str(int(len(test_labels))) + '\n')
    f.write('Accuracy: ' + str(accuracy) + '\n')
    f.write('F1_score: ' + str(f1score) + '\n')
    f.write("Meibomio selezionato % (% di 1): " + str(count1/count2 * 100) + '\n')
    f.write("Meibomio perso %: " + str(100 - (count1/count2 * 100)) + '\n')
    f.write("Parte non corrispondente al Meibomio selezionata %: " + str(count3/count4 * 100) + '\n')
    f.write(metrics.classification_report(test_labels, predict) + '\n')
    f.close()

if __name__ == "__main__":
    main()
