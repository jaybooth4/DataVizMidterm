import pickle
from pyspark import SparkContext


def saveData(data, fName):
    with open("backendOutput/" + fName + '.pkl', 'wb') as f:
        pickle.dump(data, f)


def loadData(fName):
    with open("backendOutput/" + fName + '.pkl', 'rb') as f:
        return pickle.load(f)


def parallelizeData(data, sc, size=None):
    ''' Return an RDD of string '''
    if size:
        data = data[:size]
    return sc.parallelize(data)


def createSparkContext(nproc=4):
    return SparkContext("local[" + str(nproc) + "]", 'Data Visualization Preprocessing')
