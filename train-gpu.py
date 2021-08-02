from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time
from utils import matrix
from utils import save_read_classifier as srcl
import thundersvm
from sklearn.model_selection import GridSearchCV
import os

def main():
    # gpu 1 without torch, gpu 0 with torch (using nvidia-smi in terminal the gpus are inverted)
    selectGpu = 0
    # selectGpu = 1
    if selectGpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print('start')

    datasetNotCreate = False
    isUpperGland = True
    useBestParam = False
    if datasetNotCreate:
        if isUpperGland:
            cov_matrix, labels = matrix.read_matrix('Matrices/matrix_svm_All_10x10_coordinate_Up')
        else:
            cov_matrix, labels = matrix.read_matrix('Matrices/matrix_svm_All_10x10_coordinate_Low')
        train_cov, test_cov, train_labels, test_labels = train_test_split(
        cov_matrix, labels[0], random_state=None, shuffle=False,
        test_size=0.3) # 70% training and 30% test

        if isUpperGland:
            matrix.create_matrix(train_cov, train_labels, 'Dataset/train_up')
            matrix.create_matrix(test_cov, test_labels, 'Dataset/test_up')
        else:
            matrix.create_matrix(train_cov, train_labels, 'Dataset/train_low')
            matrix.create_matrix(test_cov, test_labels, 'Dataset/test_low')

    if datasetNotCreate == False:
        if isUpperGland:
            train_cov, train_labels = matrix.read_matrix('Dataset/train_up')
        else:
            train_cov, train_labels = matrix.read_matrix('Dataset/train_low')
        train_labels = train_labels[0]

    time_start = time.time()
    print(time_start)

    pca = PCA(.995, whiten=True)
    if useBestParam:
        params_grid = {'C': [0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1]}
        clf = GridSearchCV(thundersvm.SVC(probability=True, verbose=True), params_grid, verbose=True)
    else:
        clf = thundersvm.SVC(probability=True, kernel='rbf', verbose=True)
    clf = make_pipeline(StandardScaler(), pca, clf)
    clf.fit(train_cov, train_labels)

    if isUpperGland:
        srcl.saveSVM(clf, "classifier/classifier_All_10x10_coordinate_Up")
    else:
        srcl.saveSVM(clf, "classifier/classifier_All_10x10_coordinate_Low")

    time_end_fit = time.time()
    tmp = time_end_fit - time_start
    print(tmp)
  
  
if __name__ == "__main__":
    main()

