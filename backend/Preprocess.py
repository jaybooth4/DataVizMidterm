import sys
import math
from pyspark import SparkContext
from operator import add
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
from .Tokenize import tokenizerFactory
from .NGram import convertNGram
from operator import add

def tokenizeCorpus(dataRdd, tokenizer):
    return dataRdd.map(lambda doc: tokenizer.tokenize(doc))

def removeOutliers(dataRdd, minDf=1, maxDf=.5):
    dataSize = dataRdd.count()
    maxDfCount = dataSize * maxDf
    # Identify outliers present in too few or too many files
    outliers = dataRdd.map(lambda tokens: list(set(tokens)))\
                      .flatMap(lambda entry: [(word, 1) for word in entry[1]])\
                      .reduceByKey(add)\
                      .filter(lambda entry: entry[1] > minDf and entry[1] < maxDfCount)\
                      .map(lambda entry: entry[0])\
                      .collect()
    # Filter data 
    return dataRdd.map(lambda tokens: [token for token in tokens if token not in outliers])

def readTrainData(textOnly=True):
    if textOnly:
        return fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    else:
        return fetch_20newsgroups(subset='train', download_if_missing=True)


def readTestData(textOnly=True):
    if textOnly:
        return fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    else:
        return fetch_20newsgroups(subset='test', download_if_missing=True)


def parallelizeData(data, sc, size=None):
    ''' Return an RDD of string '''
    if size:
        data = data[:size]
    return sc.parallelize(data)

def preprocess(tokenizerType, size, nproc=4, ngram=False):
    ''' Returns corpus list of lists of tokens + list with labels '''
    sc = SparkContext("local["+ str(nproc) +"]", 'Data Visualization Preprocessing')

    rawData = readTrainData()
    corpusRdd = parallelizeData(rawData.data, sc, size)
    tokenizer = tokenizerFactory(tokenizerType)
    tokenizedCorpus = tokenizeCorpus(corpusRdd, tokenizer).cache(nproc)
    cleanCorpus = removeOutliers(tokenizedCorpus).collect()
    tokenizedCorpus.unpersist()
    return cleanCorpus, rawData.target[:size]