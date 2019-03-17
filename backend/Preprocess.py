from operator import add

from sklearn.datasets import fetch_20newsgroups

from .NGram import convertNGram
from .Tokenize import tokenizerFactory
from .Util import parallelizeData, saveData


def tokenizeCorpus(dataRdd, tokenizer):
    '''
    Tokenize the dataset
    input: Rdd of strings, tokenizer object
    output: Rdd of list of strings
    '''
    return dataRdd.map(lambda doc: tokenizer.tokenize(doc))


def removeOutliers(dataRdd, minDf=5, maxDf=.25):
    '''
    Remove outliers in the
    input: Rdd of list of strings, min number, and max percentage of documents a token must occur in
    output: Rdd of list of strings
    '''
    dataSize = dataRdd.count() * 1.0
    maxDfCount = dataSize * maxDf
    # Identify outliers present in too few or too many files
    outliers = dataRdd.map(lambda tokens: list(set(tokens)))\
                      .flatMap(lambda tokens: [(word, 1.0) for word in tokens])\
                      .reduceByKey(add)\
                      .filter(lambda entry: entry[1] < minDf or entry[1] > maxDfCount)\
                      .map(lambda entry: entry[0])\
                      .collect()
    outliers = set(outliers)
    # Filter data
    return dataRdd.map(lambda tokens: list(filter(lambda token: token not in outliers, tokens)))


def readData(subset="all", textOnly=True):
    '''
    Get the dataset
    input: what subset is desired, if headers footers and quotes should be removed
    output: dataset object
    '''
    if textOnly:
        return fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    else:
        return fetch_20newsgroups(subset=subset, download_if_missing=True)


def preprocess(tokenizerType, sc, size=None, ngram=False, save=False, name=""):
    ''' 
    Preprocesses the dataset
    input: type of tokenizer, sparkcontext, size of dataset to get, to use ngrams, save file?, file name
    output: list of list of tokens, list of labels
    '''
    rawData = readData()
    corpusRdd = parallelizeData(rawData.data, sc, size)
    tokenizer = tokenizerFactory(tokenizerType)
    tokenizedCorpus = tokenizeCorpus(corpusRdd, tokenizer).cache()
    cleanCorpus = removeOutliers(tokenizedCorpus).collect()
    tokenizedCorpus.unpersist()
    labels = [rawData.target_names[i] for i in rawData.target[:size]]
    if ngram:
        cleanCorpus = convertNGram(cleanCorpus)
    if save:
        saveData([cleanCorpus, labels], "preprocess-" + name)
    return cleanCorpus, labels
