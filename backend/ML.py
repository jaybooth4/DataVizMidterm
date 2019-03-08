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
from sklearn.metrics import adjusted_rand_score


def trainLDA(docRep, dictionary, save=False, name='ldamodel'):
    ''' Function to train and return an ldamodel. Expects a sparse matrix as input '''
    ldamodel = LdaModel(Sparse2Corpus(docRep, documents_columns=False),
                        num_topics=20, id2word=dictionary)
    if save:
        tempFile = "../results/" + name
        # To load the model use: lda = LdaModel.load(temp_file)
        ldamodel.save(tempFile)
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