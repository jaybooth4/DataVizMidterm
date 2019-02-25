#!/usr/bin/env python
# coding: utf-8


import pickle
from abc import ABCMeta, abstractmethod

import numpy as np
import spacy
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import Doc2Vec, LdaModel
from gensim.models.phrases import Phrases
from gensim.models.tfidfmodel import TfidfModel
from gensim.test.utils import datapath, get_tmpfile
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import adjusted_rand_score
from spacy.tokenizer import Tokenizer

download('stopwords')
download('punkt')

# Preprocessing Base Class
class Preprocess(metaclass=ABCMeta):
    def __init__(self, ngramCount=1, min_df=1, max_df=.5):
        self.ngram_range = (1, ngramCount)
        self.max_df = max_df
        self.min_df = min_df

    @abstractmethod
    def preprocess(self, data):
        pass

# Preprocessor using NLTK
class NLTKPreProcess(Preprocess):
    def __init__(self, ngramCount=1, max_df=.5, min_df=1):
        Preprocess.__init__(self, ngramCount, min_df, max_df)
        self.stop = set(stopwords.words('english'))

    def filterWords(self, word):
        return word not in self.stop and word.isalpha()

    def customTokenizer(self, doc):
        words = [word for sentence in sent_tokenize(doc) for word in word_tokenize(sentence)]
        return(list(filter(lambda word: self.filterWords(word), words)))

    def preprocess(self, data, save=False):
        nltkCountVectorizer = CountVectorizer(tokenizer=self.customTokenizer, ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df)
        bow = nltkCountVectorizer.fit_transform(data)
        tfidf = TfidfTransformer().fit_transform(bow)
        id2word = dict((id, word) for word, id in nltkCountVectorizer.vocabulary_.items())
        if save:
            pickleData([bow, tfidf, id2word], "preprocess")
        return bow, tfidf, id2word


# Preprocessor using Spacy
class SpacyPreprocess(Preprocess):
    def __init__(self, ngramCount=1, max_df=.5, min_df=1):
        Preprocess.__init__(self, ngramCount, max_df, min_df)
        self.stop = set(stopwords.words('english'))
        self.nlp = spacy.load("en", disable=['tagger', 'parser', 'ner', 'textcat'])
        self.addStopWords([])
        self.tokenizer = Tokenizer(self.nlp.vocab)

    def preprocess(self, data, save=False):            
        spacyCountVectorizer = CountVectorizer(tokenizer=self.customTokenizer, ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df)
        bow = spacyCountVectorizer.fit_transform(data)
        tfidf = TfidfTransformer().fit_transform(bow)
        id2word = dict((id, word) for word, id in spacyCountVectorizer.vocabulary_.items())
        if save:
            pickleData([bow, tfidf, id2word], "preprocess")
        return bow, tfidf, id2word

    def addStopWords(self, stopWords):
        for stopWord in stopWords:
            lexeme = self.nlp.vocab[stopWord]
            lexeme.is_stop = True
            
    def filterWords(self, word):
        return word.is_alpha and not word.is_stop

    def convertWords(self, word):
        return word.lemma_.lower()

    def customTokenizer(self, doc):
        docTokens = self.tokenizer(doc)
        return list(map(self.convertWords, filter(self.filterWords, docTokens)))

def trainLDA(docRep, dictionary, save=False):
    ''' Function to train and return an ldamodel. Expects a sparse matrix as input '''
    ldamodel = LdaModel(Sparse2Corpus(docRep), num_topics=20, id2word=dictionary)
    if save:
        tempFile = datapath("../results/ldaModel")
        ldamodel.save(tempFile) # To load the model use: lda = LdaModel.load(temp_file)
    return ldamodel

def convertToDoc2Vec(bow, id2word, labels):
    ''' Function to convert data to proper format for Doc2Vec '''
    doc2VecData = []
    for docIndex, doc in enumerate(bow):
        doc2VecData.append(([id2word.get(id) for id in doc], labels[docIndex]))
    return doc2VecData

def trainDoc2Vec(data, save=False):
    ''' Train a Doc2Vec model '''
    model = Doc2Vec()
    model.build_vocab(data)
    if save:
        fname = get_tmpfile("../results/doc2vecModel")
        model.save(fname)
    return model

def clusterData(clusterData, labels, save=False):
    ''' Run kmeans on given data '''
    kmeans = KMeans(n_clusters=20).fit(clusterData)
    if save:
        pickleData(kmeans.labels_, "kmeans")
    return adjusted_rand_score(kmeans.labels_, labels)

def clusterDataMiniBatch(clusterData, labels, save=False):
    ''' Run minibatch kmeans on given data for faster performance '''
    kmeans = MiniBatchKMeans(n_clusters=20).fit(clusterData)
    if save:
        pickleData(kmeans.labels_, "mbKmeans")
    return adjusted_rand_score(kmeans.labels_, labels)

def pickleData(data, fName):
    with open("../results/" + fName + '.pkl', 'wb') as f:
        pickle.dump(data, f)

def loadData(data, fName):
    with open("../results/" + fName + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    dataSize = 100

    newsTrain = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    newsTest = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    preprocess = NLTKPreProcess() # SpacyPreprocess()
    bow, tfidf, id2word = preprocess.preprocess(newsTrain.data[:dataSize], True)
    ldaModelBOW = trainLDA(bow, id2word)
    ldaModelTFIDF = trainLDA(tfidf, id2word)
    clusterData(bow, newsTrain.target[:dataSize])
    clusterData(tfidf, newsTrain.target[:dataSize])

if __name__ == "__main__":
    # execute only if run as a script
    main()
