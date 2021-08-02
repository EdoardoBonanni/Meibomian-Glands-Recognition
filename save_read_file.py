import pickle

def saveFile(obj, name):
    with open(name + ".file", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadFile(name):
    with open(name + ".file", "rb") as f:
        obj = pickle.load(f)
        return obj