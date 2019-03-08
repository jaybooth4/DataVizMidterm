import sys
import math
from operator import add
from sklearn.datasets import fetch_20newsgroups
from .Tokenize import tokenizerFactory
from .NGram import convertNGram
from .Util import parallelizeData, saveData
from operator import add


def tokenizeCorpus(dataRdd, tokenizer):
    return dataRdd.map(lambda doc: tokenizer.tokenize(doc))


def removeOutliers(dataRdd, minDf=2, maxDf=.5):
    dataSize = dataRdd.count()
    maxDfCount = dataSize * maxDf
    # Identify outliers present in too few or too many files
    outliers = dataRdd.map(lambda tokens: list(set(tokens)))\
                      .flatMap(lambda tokens: [(word, 1) for word in tokens])\
                      .reduceByKey(add)\
                      .filter(lambda entry: entry[1] < minDf or entry[1] > maxDfCount)\
                      .map(lambda entry: entry[0])\
                      .collect()
    outliers = set(outliers)
    # Filter data
    return dataRdd.map(lambda tokens: list(filter(lambda token: token not in outliers, tokens)))


def readData(subset="train", textOnly=True):
    if textOnly:
        return fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    else:
        return fetch_20newsgroups(subset=subset, download_if_missing=True)


def preprocess(tokenizerType, size, sc, save=False, ngram=False):
    ''' Returns corpus list of lists of tokens + list with labels '''
    rawData = readData()
    corpusRdd = parallelizeData(rawData.data, sc, size)
    tokenizer = tokenizerFactory(tokenizerType)
    tokenizedCorpus = tokenizeCorpus(corpusRdd, tokenizer).cache()
    cleanCorpus = removeOutliers(tokenizedCorpus).collect()
    tokenizedCorpus.unpersist()
    labels = rawData.target[:size]
    if save:
        saveData([cleanCorpus, labels], "preprocess")
    return cleanCorpus, labels
