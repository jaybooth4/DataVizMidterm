from .Util import parallelizeData, saveData
import uuid
from operator import add
import math

def docSize(rdd):
    return rdd.map(lambda doc: len(doc))
              
def docCharSize(rdd):
    return rdd.map(lambda doc: sum(map(lambda word: len(word), doc)))

def termFrequency(rdd):
    return rdd.flatMap(lambda tokens: tokens) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(add)

def documentFrequency(rdd):
    return rdd.map(lambda doc: (str(uuid.uuid4()), doc))\
                    .flatMapValues(lambda doc: doc)\
                    .distinct() \
                    .map(lambda entry: (entry[1], 1.0)) \
                    .reduceByKey(add)

def inverseDocumentFrequency(docFrequency):
    ''' Expects input RDD of the format (word, count) '''
    docFrequency.cache()
    corpusCount = docFrequency.count()
    idf = docFrequency.mapValues(lambda count: math.log(corpusCount / count))
    docFrequency.unpersist()
    return idf

def getStats(data, sc, save=False, name=""):
    dataRdd = parallelizeData(data, sc).cache()
    size = docSize(dataRdd).collect()
    charSize = docCharSize(dataRdd).collect()
    tf = termFrequency(dataRdd).collect()
    dfRdd = documentFrequency(dataRdd)
    idf = inverseDocumentFrequency(dfRdd).collect()
    if save:
        saveData([size, charSize, tf, idf], "stats-" + name)
    return size, charSize, tf, idf