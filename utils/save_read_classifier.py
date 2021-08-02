import pickle

def saveSVM(obj, name):
    with open(name + ".file", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadSVM(name):
    with open(name + ".file", "rb") as f:
        obj = pickle.load(f)
        return obj