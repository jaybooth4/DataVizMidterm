from collections import Counter
from .Util import parallelizeData
import uuid 
import math
from operator import add


def bagOfWordsFormat(dataRdd):
    return dataRdd.map(lambda tokens: list(Counter(tokens).items()))

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

def TFIDF(termFreqRdd, idfRdd):
    return termFreqRdd.join(idfRdd) \
        .mapValues(lambda tfidf: tfidf[0] * tfidf[1])

def TFIDFFormat(data):
    tfRdd = termFrequency(data)
    dfRdd = documentFrequency(data)
    idfRdd = inverseDocumentFrequency(dfRdd)
    return TFIDF(tfRdd, idfRdd)

def doc2VecFormat(corpus, labels):
    return [[corpus[i], labels[i]] for i in range(len(corpus))]        

def getEmbeddings(data, labels, sc):
    dataRdd = parallelizeData(data, sc).cache()
    bow = bagOfWordsFormat(dataRdd).collect()
    tfidf = TFIDFFormat(dataRdd).collect()
    doc2Vec = doc2VecFormat(data, labels)
    dataRdd.unpersist()
    return bow, tfidf, doc2Vec