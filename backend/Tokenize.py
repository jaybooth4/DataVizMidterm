#!/usr/bin/env python
# coding: utf-8

from abc import ABCMeta, abstractmethod

import spacy
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.tokenizer import Tokenizer

# download('stopwords')
# download('punkt')

# Preprocessing Base Class
class Preprocess(metaclass=ABCMeta):
    def __init__(self, useNgram=False, minDf=1, maxDf=.5):
        self.useNgram = useNgram
        self.minDf = minDf
        self.maxDf = maxDf

    @abstractmethod
    def tokenize(self, data):
        pass

    @abstractmethod
    def addStopWords(self, stopWords):
        pass

# Preprocessor using NLTK
class NLTKPreprocess(Preprocess):
    def __init__(self, useNgram=False, minDf=1, maxDf=.5):
        Preprocess.__init__(self, useNgram, minDf, maxDf)
        self.stop = set(stopwords.words('english'))


    def addStopWords(self, stopWords):
        for stopWord in stopWords:
            self.stop.add(stopWord)

    def filterWord(self, word):
        return word not in self.stop and word.isalpha()

    def convertWord(self, word):
        return word.lower()

    def tokenize(self, doc):
        words = [word for sentence in sent_tokenize(doc) for word in word_tokenize(sentence)]
        return list(map(self.convertWord, filter(self.filterWord, words)))


# Preprocessor using Spacy
class SpacyPreprocess(Preprocess):
    def __init__(self, ngramCount=1, minDf=1, maxDf=.5):
        Preprocess.__init__(self, ngramCount, maxDf, minDf)
        self.nlp = spacy.load("en", disable=['tagger', 'parser', 'ner', 'textcat'])
        self.tokenizer = Tokenizer(self.nlp.vocab)

    def addStopWords(self, stopWords):
        for stopWord in stopWords:
            lexeme = self.nlp.vocab[stopWord]
            lexeme.is_stop = True
            
    def filterWords(self, word):
        return word.is_alpha and not word.is_stop

    def convertWords(self, word):
        return word.lemma_.lower()

    def tokenize(self, doc):
        docTokens = self.tokenizer(doc)
        return list(map(self.convertWords, filter(self.filterWords, docTokens)))

def tokenizerFactory(tokenizerType, ngramCount=1, minDf=1, maxDf=.5):
    if tokenizerType == "NLTK":
        return NLTKPreprocess(ngramCount, minDf, maxDf)
    elif tokenizerType == "SPACY":
        return SpacyPreprocess(ngramCount, minDf, maxDf)
    else:
        raise NameError("Unsupported tokenizer type.")