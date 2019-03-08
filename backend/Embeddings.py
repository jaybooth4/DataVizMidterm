import sys
import math
from pyspark import SparkContext
from operator import add
from sklearn.datasets import fetch_20newsgroups

def toLowerCase(s):
    return s.lower()

def stripNonAlpha(s):
    return ''.join([c for c in s if c.isalpha()])

def termFrequency(rdd):
    return rdd.flatMap(lambda (fileName, text): text.split()) \
        .map(lambda word: (toLowerCase(stripNonAlpha(word)), 1)) \
        .filter(lambda (word, count): word is not "") \
        .reduceByKey(add)

def documentFrequency(rdd, maxFreq, minFreq):
    fileCount = rdd.count()
    return rdd.flatMapValues(lambda text: text.split()) \
        .map(lambda (fileName, word): (fileName, toLowerCase(stripNonAlpha(word)))) \
        .distinct() \
        .filter(lambda (fileName, word): word != "") \
        .map(lambda (fileName, word): (word, 1.0)) \
        .reduceByKey(add) \
        .filter(lambda (word, count): count > min and count < max) \

def inverseDocumentFrequency(docFrequency, corpusCount):
        return docFrequency.map(lambda (word, count): (word, math.log(corpusCount / count)))

def TFIDF(termFreqRdd, idfRdd):
    return termFreqRdd.join(idfRdd) \
        .mapValues(lambda (tf, idf): tf * idf)

def readData(size):
    return fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), download_if_missing=True)

def convertData(data):
    dataRdd = sc.parallelize(data)
    return data.map(lambda news: (news.target, news.data))

def preprocessFactory(preprocessType):

def preprocess(preprocessType, size):
    rawData = readData(size)
    sc = SparkContext(args.master, 'Data Visualization Preprocessing')
    preprocess = preprocessFactory(preprocessType)
    corpusRdd = convertData(data, sc)

