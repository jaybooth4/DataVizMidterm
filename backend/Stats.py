import math
import uuid
from operator import add

from .Util import parallelizeData, saveData


def docSize(rdd):
    '''
    Get size of the documents
    '''
    return rdd.map(lambda doc: len(doc))


def docCharSize(rdd):
    '''
    Get number of chars for each of the documents
    '''
    return rdd.map(lambda doc: sum(map(lambda word: len(word), doc)))


def termFrequency(rdd):
    '''
    Get number of times each token appears
    '''
    return rdd.flatMap(lambda tokens: tokens) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(add)


def documentFrequency(rdd):
    '''
    Get number of documents each token appears in
    '''
    return rdd.map(lambda doc: (str(uuid.uuid4()), doc))\
        .flatMapValues(lambda doc: doc)\
        .distinct() \
        .map(lambda entry: (entry[1], 1.0)) \
        .reduceByKey(add)


def inverseDocumentFrequency(docFrequency):
    ''' 
    Generate inverse document frequency
    input: RDD of the format (word, count) 
    '''
    docFrequency.cache()
    corpusCount = docFrequency.count()
    idf = docFrequency.mapValues(lambda count: math.log(corpusCount / count))
    docFrequency.unpersist()
    return idf


def getStats(data, sc, save=False, name=""):
    ''' 
    Get docsize, charsize, term frequency, and inverse document freqency for data
    input: list of list of tokens, spark context
    '''
    dataRdd = parallelizeData(data, sc).cache()
    size = docSize(dataRdd).collect()
    charSize = docCharSize(dataRdd).collect()
    tf = termFrequency(dataRdd).collect()
    dfRdd = documentFrequency(dataRdd)
    idf = inverseDocumentFrequency(dfRdd).collect()
    if save:
        saveData([size, charSize, tf, idf], "stats-" + name)
    return size, charSize, tf, idf
