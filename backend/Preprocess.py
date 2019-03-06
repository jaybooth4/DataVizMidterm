#!/usr/bin/env python
# coding: utf-8


import pickle
from abc import ABCMeta, abstractmethod

import spacy
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

def pickleData(data, fName):
    with open("../results/" + fName + '.pkl', 'wb') as f:
        pickle.dump(data, f)

def loadData(fName):
    with open("../results/" + fName + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    dataSize = 100

    newsTrain = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    newsTest = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    preprocess = NLTKPreProcess() # SpacyPreprocess()
    bow, tfidf, id2word = preprocess.preprocess(newsTrain.data[:dataSize], True)

if __name__ == "__main__":
    # execute only if run as a script
    main()
