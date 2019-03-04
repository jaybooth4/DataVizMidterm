# def pickleData(data, fName):
#     with open("../results/" + fName + '.pkl', 'wb') as f:
#         pickle.dump(data, f)

import pickle

def load_data(fName):
    with open("../backendOutput/" + fName + '.pkl', 'rb') as f:
        return pickle.load(f)
