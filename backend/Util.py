import pickle

def pickleData(data, fName):
    with open("../results/" + fName + '.pkl', 'wb') as f:
        pickle.dump(data, f)

def loadData(fName):
    with open("../results/" + fName + '.pkl', 'rb') as f:
        return pickle.load(f)