from scipy.io import savemat
from scipy.io import loadmat

def create_matrix(cov_matrix, labels, name):
    dict = {"cov_matrix": cov_matrix, "labels": labels}
    savemat(name + ".mat", dict)

def read_matrix(name):
    mat = loadmat(name + ".mat")
    return mat['cov_matrix'], mat['labels']

def create_matrix_mask(matrix, name):
    dict = {"matrix": matrix}
    savemat(name + ".mat", dict)

def read_matrix_mask(name):
    mat = loadmat(name + ".mat")
    return mat['matrix']
