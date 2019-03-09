#!/usr/bin/env python
# coding: utf-8

import pickle

from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import Doc2Vec, ldamulticore
from gensim.models.tfidfmodel import TfidfModel
from gensim.test.utils import datapath, get_tmpfile
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import normalized_mutual_info_score
from .Util import saveData


def trainLDA(docRep, dictionary, save=False, name=""):
    ''' Function to train and return an ldamodel. Expects a sparse matrix as input '''
    corpus = Sparse2Corpus(docRep, documents_columns=False)
    ldamodel = ldamulticore.LdaMulticore(
        corpus, num_topics=20, id2word=dictionary, workers=4)
    if save:
        saveData(ldamodel, 'ldamodel-' + name)
    return ldamodel


def trainDoc2VecByTopic(data, minCount=1, vecSize=20, save=False, name=""):
    ''' Train a Doc2Vec model to get representations of topics, expects [(tokens, topic)] '''
    doc2VecModel = Doc2Vec(data,
                           vector_size=vecSize,
                           minCount=minCount,
                           workers=4)
    if save:
        saveData(doc2VecModel, 'doc2vec-' + name)
    return doc2VecModel


def trainDoc2Vec(corpus, minCount=1, vecSize=20, save=False, name=""):
    ''' Train a Doc2Vec model, expects corpus '''
    data = [TaggedDocument(words=doc, tags=[idx]) for idx, doc in enumerate(corpus)]
    doc2VecModel = Doc2Vec(data,
                           vector_size=vecSize,
                           minCount=minCount,
                           workers=4)
    if save:
        saveData(doc2VecModel, 'doc2vec-' + name)
    return doc2VecModel


def clusterDataMiniBatch(clusterData, labels, save=False, name=""):
    ''' Run minibatch kmeans on given data for faster performance '''
    kmeans = MiniBatchKMeans(n_clusters=20).fit(clusterData)
    if save:
        saveData(kmeans.labels_, "kmeans-" + name)
    return kmeans.labels_, normalized_mutual_info_score(kmeans.labels_, labels)


def getLDARep(ldaModel, docRep, save=False, name=""):
    ''' Convert doc representation to lda output '''
    corpus = Sparse2Corpus(docRep, documents_columns=False)
    converted = ldaModel.get_document_topics(corpus, minimum_probability=0.0)
    rep = [list(map(lambda topic: topic[1], converted[i]))
           for i in range(len(corpus))]
    if save:
        saveData(rep, name)
    return rep


def getDoc2VecRep(doc2VecModel, save=False, name=""):
    ''' Convert doc2vec to vector representation '''
    rep = [doc2VecModel.docvecs[i] for i in range(len(doc2VecModel.docvecs))]
    if save:
        saveData(rep, name)
    return rep
