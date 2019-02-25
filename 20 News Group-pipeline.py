#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning and ML

# Download Data
from sklearn.datasets import fetch_20newsgroups
from nltk import download
download('stopwords')
download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from abc import ABCMeta, abstractmethod
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel
import spacy
from gensim.models.phrases import Phrases
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from sklearn.cluster import KMeans
import numpy as np
from spacy.tokenizer import Tokenizer
from sklearn.metrics import adjusted_rand_score
from gensim.models import Doc2Vec

# ## Preprocessing Classes

class Preprocess(metaclass=ABCMeta):
    def __init__(self, ngramCount=1):

    @abstractmethod
    def preprocess(self, data):
        pass

class NLTKPreProcess(Preprocess):
    def __init__(self, ngramCount=1):
        Preprocess.__init__(self, ngramCount)
        self.stop = set(stopwords.words('english'))
        self.max_df = .5
        self.min_df = 5
        self.ngram_range = (1, ngramCount)

    def filterWords(self, word):
        return word not in self.stop and word.isalpha()

    def customTokenizer(self, doc):
        words = [word for sentence in sent_tokenize(doc) for word in word_tokenize(sentence)]
        return(list(filter(lambda word: self.filterWords(word), words)))

    def preprocess(self, data):
        nltkCountVectorizer = CountVectorizer(tokenizer=self.customTokenizer, ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df)
        bow = nltkCountVectorizer.fit_transform(data)
        tfidf = TfidfTransformer().fit_transform(bow)
        id2word = dict((id, word) for word, id in nltkCountVectorizer.vocabulary_.items())
        return bow, tfidf, id2word


class SpacyPreprocess(Preprocess):
    def __init__(self, ngramCount=1):
        Preprocess.__init__(self, ngramCount)
        self.stop = set(stopwords.words('english'))
        self.max_df = .5
        self.min_df = 5
        self.ngram_range = (1, ngramCount)
        self.nlp = spacy.load("en", disable=['tagger', 'parser', 'ner', 'textcat'])
        self.addStopWords([])
        self.tokenizer = Tokenizer(self.nlp.vocab)

    def preprocess(self, data):            
        spacyCountVectorizer = CountVectorizer(tokenizer=self.customTokenizer, ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df)
        bow = spacyCountVectorizer.fit_transform(data)
        tfidf = TfidfTransformer().fit_transform(bow)
        id2word = dict((id, word) for word, id in nltkCountVectorizer.vocabulary_.items())
        return bow, tfidf, dictionary

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

def trainLDA(docRep, dictionary):
    ldamodel = LdaModel(Sparse2Corpus(docRep), num_topics=20, id2word=id2word)

def convertToDoc2Vec(bow, id2word, labels):
    doc2VecData = []
    for docIndex, doc in enumerate(bow):
        doc2VecData.append(([id2word.get(id) for id in doc], labels[docIndex]))
    return doc2VecData

def trainDoc2Vec(data):
    model = Doc2Vec()
    model.build_vocab(data)
    return model

def clusterData(clusterData, labels):
    kmeans = KMeans(n_clusters=20).fit(clusterData)

    # Similarity metric between groups
    score = adjusted_rand_score(kmeans.labels_, labels)
    print(score)

def clusterDataMiniBatch(clusterData, labels):
    kmeans = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose).fit(clusterData)
    score = adjusted_rand_score(kmeans.labels_, labels)
    print(score)

def main():
    newsTrain = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    newsTest = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    preprocess = NLTKPreProcess() # SpacyPreprocess()
    bow, tfidf, id2word = preprocess.preprocess(newsTrain[:10])
    ldaModelBOW = trainLDA(bow, id2word)
    ldaModelTFIDF = trainLDA(tfidf, id2word)
    clusterData(bow, newsTrain.target)
    clusterData(tfidf, newsTrain.target)

if __name__ == "__main__":
    # execute only if run as a script
    main()