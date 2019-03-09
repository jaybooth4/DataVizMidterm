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
    # sc = createSparkContext()


    # Preprocess
    # corpus, labels = preprocess("NLTK", sc, save=True, name="nltk")
    # corpus, labels = loadData("preprocess-nltk")

    # corpus, labels = preprocess("Spacy", sc, save=True, name="spacy")
    # corpus, labels = loadData("preprocess-spacy")


    # Embeddings
    # size, charSize, tf, idf = getStats(corpus, sc, save=True, name="nltk")
    # bow, tfidf, doc2VecFormat, id2word = getEmbeddings(corpus, labels, save=True, name="nltk")

    # size, charSize, tf, idf = getStats(corpus, sc, save=True, name="spacy")
    # bow, tfidf, doc2VecFormat, id2word = getEmbeddings(corpus, labels, save=True, name="spacy")


    #ML
    # bow, tfidf, doc2VecFormat, id2word = loadData("embeddings-nltk")
    # ldaModelBow = trainLDA(bow, id2word, save=True, name="nltk-bow")
    # print("lda bow")
    # ldaModelTfidf = trainLDA(tfidf, id2word, save=True, name="nltk-tfidf")
    # print("lda tfidf")
    # Doc2VecModel = trainDoc2Vec(doc2VecFormat, save=True, name="nltk")
    # print("doc2vec")

    bow, tfidf, doc2VecFormat, id2word = loadData("embeddings-spacy")
    ldaModelBow = trainLDA(bow, id2word, save=True, name="spacy-bow")
    print("lda bow")
    ldaModelTfidf = trainLDA(tfidf, id2word, save=True, name="spacy-tfidf")
    print("lda tfidf")
    Doc2VecModel = trainDoc2Vec(doc2VecFormat, save=True, name="spacy")
    print("doc2vec")
    
    # kmeansLabels, nmi = clusterData(bow, labels, save=True)
    print("success")


    # Visualizations
    # graphTFVsImp()

    # graphKMeansClusters(bow, labels, True)
    

if __name__ == "__main__":
    # execute only if run as a script
    main()