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
    def __init__(self):
        pass
        
    @abstractmethod
    def tokenize(self, data):
        pass

    @abstractmethod
    def addStopWords(self, stopWords):
        pass

# Preprocessor using NLTK
class NLTKPreprocess(Preprocess):
    def __init__(self):
        Preprocess.__init__(self)
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
        return list(filter(self.filterWord, map(self.convertWord, words)))


# Preprocessor using Spacy
class SpacyPreprocess(Preprocess):
    def __init__(self):
        Preprocess.__init__(self)
        self.nlp = spacy.load("en", disable=['tagger', 'parser', 'ner', 'textcat'])
        self.addStopWords(self.nlp.Defaults.stop_words)
        self.tokenizer = Tokenizer(self.nlp.vocab)

    def addStopWords(self, stopWords):
        for stopWord in stopWords:
            for w in (stopWord, stopWord[0].upper() + stopWord[1:], stopWord.upper()):
                lexeme = self.nlp.vocab[w]
                lexeme.is_stop = True
            
    def filterWords(self, word):
        return word.is_alpha and not word.is_stop

    def convertWords(self, word):
        return word.lemma_.lower()

    def tokenize(self, doc):
        docTokens = self.tokenizer(doc)
        return list(map(self.convertWords, filter(self.filterWords, docTokens)))

def tokenizerFactory(tokenizerType):
    if tokenizerType == "NLTK":
        return NLTKPreprocess()
    elif tokenizerType == "Spacy":
        return SpacyPreprocess()
    else:
        raise NameError("Unsupported tokenizer type.")