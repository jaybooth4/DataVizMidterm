#!/usr/bin/env python
# coding: utf-8

import pickle

from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import Doc2Vec, LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.test.utils import datapath, get_tmpfile
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import normalized_mutual_info_score
from .Util import saveData

def trainLDA(docRep, dictionary, save=False, name=""):
    ''' Function to train and return an ldamodel. Expects a sparse matrix as input '''
    ldamodel = LdaModel(Sparse2Corpus(docRep, documents_columns=False),
                        num_topics=20, id2word=dictionary)
    if save:
        saveData(ldamodel, 'ldamodel-' + name)
    return ldamodel


def trainDoc2Vec(data, save=False, name=""):
    ''' Train a Doc2Vec model, expects [(tokens, label)] '''
    doc2VecModel = Doc2Vec()
    doc2VecModel.build_vocab(data)
    if save:
        saveData(doc2VecModel, 'doc2vec-' + name)
    return doc2VecModel


def clusterData(clusterData, labels, save=False, name=""):
    ''' Run kmeans on given data '''
    kmeans = KMeans(n_clusters=20).fit(clusterData)
    if save:
        saveData(kmeans.labels_, "kmeans-" + name)
    return kmeans.labels_, normalized_mutual_info_score(kmeans.labels_, labels)


def clusterDataMiniBatch(clusterData, labels, save=False, name=""):
    ''' Run minibatch kmeans on given data for faster performance '''
    kmeans = MiniBatchKMeans(n_clusters=20).fit(clusterData)
    if save:
        saveData(kmeans.labels_, "kmeansmb-" + name)
    return kmeans.labels_, normalized_mutual_info_score(kmeans.labels_, labels)