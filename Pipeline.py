from backend.Embeddings import getEmbeddings
from backend.Preprocess import preprocess
from backend.Util import createSparkContext, loadData
from backend.ML import trainLDA, trainDoc2Vec, clusterData, clusterDataMiniBatch
from backend.Stats import getStats
from frontend.termfrequency_vs_importance_grapher import graphTFVsImp
from frontend.kmeansclustering_grapher import graphKMeansClusters
from sklearn.datasets import fetch_20newsgroups
from gensim.matutils import Sparse2Corpus
import argparse
import numpy as np


def main():
    sc = createSparkContext()
    corpus, labels = preprocess("Spacy", 100, sc, save=True)
    # corpus, labels = loadData("preprocess")
    size, charSize, tf, idf = getStats(corpus, sc, save=True)
    # bow, tfidf, doc2VecFormat, id2word = getEmbeddings(corpus, labels, save=True)
    # bow, tfidf, doc2VecFormat, id2word = loadData("embeddings")
    # ldaModel = trainLDA(bow, id2word, save=True)
    # Doc2VecModel = trainDoc2Vec(doc2VecFormat, save=True)
    # kmeansLabels, nmi = clusterData(bow, labels, save=True)
    print("success")


    # Visualizations
    # graphTFVsImp()

    # graphKMeansClusters(bow, labels, True)
    

if __name__ == "__main__":
    # execute only if run as a script
    main()