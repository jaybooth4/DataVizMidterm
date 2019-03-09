from backend.Embeddings import getEmbeddings
from backend.Preprocess import preprocess
from backend.Util import createSparkContext, loadData
from backend.ML import trainLDA, trainDoc2Vec, clusterDataMiniBatch, getLDARep, getDoc2VecRep, trainDoc2VecByTopic
from backend.Stats import getStats
from frontend.termfrequency_vs_importance_grapher import graphTFVsImp
from frontend.kmeansclustering_grapher import graphKMeansClusters
from frontend.boxPlot import boxPlot
from frontend.ldamodel_grapher import graphLDA
from frontend.similarD2V import similarD2V
from frontend.scatterD2V import scatterD2V
from frontend.similarTopics import similarTopics
from frontend.plotNMI import plotNMI
from frontend.bestK import bestK
from sklearn.datasets import fetch_20newsgroups
from gensim.matutils import Sparse2Corpus
import argparse
import numpy as np
import time


def main():
    sc = createSparkContext()

    # Preprocess
    corpus, labels = preprocess("NLTK", sc, save=True, name="nltk")
    corpus, labels = loadData("preprocess-nltk")

    # corpus, labels = preprocess("Spacy", sc, save=True, name="spacy")
    # corpus, labels = loadData("preprocess-spacy")

    # Embeddings
    size, charSize, tf, idf = getStats(corpus, sc, save=True, name="nltk")
    bow, tfidf, doc2VecFormat, id2word = getEmbeddings(corpus, labels, save=True, name="nltk")

    # size, charSize, tf, idf = getStats(corpus, sc, save=True, name="spacy")
    # bow, tfidf, doc2VecFormat, id2word = getEmbeddings(corpus, labels, save=True, name="spacy")

    # Train models
    bow, tfidf, doc2VecFormat, id2word = loadData("embeddings-nltk")
    ldaModelBow = trainLDA(bow, id2word, save=True, name="nltk-bow")
    ldaModelTfidf = trainLDA(tfidf, id2word, save=True, name="nltk-tfidf")
    doc2VecModel = trainDoc2Vec(corpus, save=True, name="nltk")
    ldaModelBow = loadData("ldamodel-nltk-bow")
    ldaRepBow = getLDARep(ldaModelBow, bow, save=True, name="ldarep-nltk-bow")
    ldaModelTfidf = loadData("ldamodel-nltk-tfidf")
    ldaReptfidf = getLDARep(ldaModelTfidf, tfidf, save=True, name="ldarep-nltk-tfidf")
    doc2VecModel = loadData("doc2vec-nltk")
    doc2vecRep = getDoc2VecRep(doc2VecModel, save=True, name="doc2vec-rep-nltk")

    # bow, tfidf, doc2VecFormat, id2word = loadData("embeddings-spacy")
    # ldaModelBow = trainLDA(bow, id2word, save=True, name="spacy-bow")
    # ldaModelTfidf = trainLDA(tfidf, id2word, save=True, name="spacy-tfidf")
    # doc2VecModel = trainDoc2Vec(corpus, save=True, name="spacy")
    # ldaModelBow = loadData("ldamodel-spacy-bow")
    # ldaRepBow = getLDARep(ldaModelBow, bow, save=True, name="ldarep-spacy-bow")
    # ldaModelTfidf = loadData("ldamodel-spacy-tfidf")
    # ldaReptfidf = getLDARep(ldaModelTfidf, tfidf, save=True, name="ldarep-spacy-tfidf")
    # doc2VecModel = loadData("doc2vec-spacy")
    # doc2vecRep = getDoc2VecRep(doc2VecModel, save=True, name="doc2vec-rep-spacy")

    # Kmeans
    doc2vecRepNltk = loadData("doc2vec-rep-nltk")
    ldaRepNltkTfidf = loadData("ldarep-nltk-tfidf")
    ldaRepNltkBow = loadData("ldarep-nltk-bow")
    clusterDataMiniBatch(bow, labels, save=True, name="nltk-bow")
    clusterDataMiniBatch(tfidf, labels, save=True, name="nltk-tfidf")
    clusterDataMiniBatch(ldaRepNltkTfidf, labels, save=True, name="nltk-lda-tfidf")
    clusterDataMiniBatch(ldaRepNltkBow, labels, save=True, name="nltk-lda-bow")
    clusterDataMiniBatch(doc2vecRepNltk, labels, save=True, name="nltk-doc2vec")

    # doc2vecRepSpacy = loadData("doc2vec-rep-spacy")
    # ldaRepSpacyTfidf = loadData("ldarep-spacy-tfidf")
    # ldaRepSpacyBow = loadData("ldarep-spacy-bow")
    # clusterDataMiniBatch(bow, labels, save=True, name="spacy-bow")
    # clusterDataMiniBatch(tfidf, labels, save=True, name="spacy-tfidf")
    # clusterDataMiniBatch(ldaRepSpacyTfidf, labels, save=True, name="spacy-lda-tfidf")
    # clusterDataMiniBatch(ldaRepSpacyBow, labels, save=True, name="spacy-lda-bow")
    # clusterDataMiniBatch(doc2vecRepSpacy, labels, save=True, name="spacy-doc2vec")

    # Visualizations

    # Box plot
    size, charSize, tf, idf = loadData("stats-nltk")
    boxPlot(range(len(size)), size, "Document Size", fileNameOut="results/boxplot-nltk-size")
    boxPlot(range(len(size)), charSize, "Document Char Size", fileNameOut="results/boxplot-nltk-charsize")

    # size, charSize, tf, idf = loadData("stats-spacy")
    # boxPlot(range(len(size)), size, "Document Size", fileNameOut="results/boxplot-spacy-size")
    # boxPlot(range(len(size)), charSize, "Document Char Size", fileNameOut="results/boxplot-spacy-charsize")

    # TF vs Imp
    graphTFVsImp(fileNameIn='backendOutput/stats-nltk.pkl', fileNameOut='results/term-frequency-vs-importance-nltk.html')
    # graphTFVsImp(fileNameIn='backendOutput/stats-spacy.pkl', fileNameOut='results/term-frequency-vs-importance-spacy.html')

    # Kmeans
    bow, tfidf, doc2VecFormat, id2word = loadData("embeddings-nltk")

    KmeansBowlabels, _ = loadData("kmeans-nltk-bow")
    graphKMeansClusters(bow, KmeansBowlabels, True, "nltk-bow")

    KmeansTfidflabels, _ = loadData("kmeans-nltk-tfidf")
    graphKMeansClusters(tfidf, KmeansTfidflabels, True, "nltk-tfidf")

    doc2vecRep = loadData("doc2vec-rep-nltk")
    kmeansDoc2veclabels, _ = loadData("kmeans-nltk-doc2vec")
    graphKMeansClusters(doc2vecRep, kmeansDoc2veclabels, False, "doc2vec-nltk")

    ldaRepBow = loadData("ldarep-nltk-bow")
    kmeansLdaBowLabels, _ = loadData("kmeans-nltk-lda-bow")
    graphKMeansClusters(ldaRepBow, kmeansLdaBowLabels, False, "lda-nltk-bow")

    ldaRepTfidf = loadData("ldarep-nltk-tfidf")
    kmeansLdaTfidfLabels, _ = loadData("kmeans-nltk-lda-tfidf")
    graphKMeansClusters(ldaRepTfidf, kmeansLdaTfidfLabels, False, "lda-nltk-tfidf")

    # bow, tfidf, doc2VecFormat, id2word = loadData("embeddings-spacy")

    # KmeansBowlabels, _ = loadData("kmeans-spacy-bow")
    # graphKMeansClusters(bow, KmeansBowlabels, True, "spacy-bow")

    # KmeansTfidflabels, _ = loadData("kmeans-spacy-tfidf")
    # graphKMeansClusters(tfidf, KmeansTfidflabels, True, "spacy-tfidf")

    # doc2vecRep = loadData("doc2vec-rep-spacy")
    # kmeansDoc2veclabels, _ = loadData("kmeans-spacy-doc2vec")
    # graphKMeansClusters(doc2vecRep, kmeansDoc2veclabels, False, "doc2vec-spacy")

    # ldaRepBow = loadData("ldarep-spacy-bow")
    # kmeansLdaBowLabels, _ = loadData("kmeans-spacy-lda-bow")
    # graphKMeansClusters(ldaRepBow, kmeansLdaBowLabels, False, "lda-spacy-bow")

    # ldaRepTfidf = loadData("ldarep-spacy-tfidf")
    # kmeansLdaTfidfLabels, _ = loadData("kmeans-spacy-lda-tfidf")
    # graphKMeansClusters(ldaRepTfidf, kmeansLdaTfidfLabels, False, "lda-spacy-tfidf")

    # LDA
    graphLDA("nltk")
    # graphLDA("spacy")

    # D2Vec
    corpus, labels = loadData("preprocess-nltk")
    doc2VecModel = loadData("doc2vec-nltk")
    similarD2V(corpus[0], doc2VecModel, labels, 20, "nltk")
    scatterD2V(labels, 20, doc2VecModel, "nltk")

    # corpus, labels = loadData("preprocess-spacy")
    # doc2VecModel = loadData("doc2vec-spacy")
    # similarD2V(corpus[0], doc2VecModel, labels, 20, "spacy")
    # scatterD2V(labels, 20, doc2VecModel, "spacy")


    # Compare kmeans results
    _, nmiNltkBow = loadData("kmeans-nltk-bow")
    _, nmiNltkTfidf = loadData("kmeans-nltk-tfidf")
    _, nmiNltkdoc2vec = loadData("kmeans-nltk-doc2vec")
    _, nmiNltkLdaBow = loadData("kmeans-nltk-lda-bow")
    _, nmiNltkLdaTfidf = loadData("kmeans-nltk-lda-tfidf")
    plotNMI(nmiNltkBow, nmiNltkTfidf, nmiNltkdoc2vec, nmiNltkLdaBow, nmiNltkLdaTfidf, "NLTK")

    # _, nmiSpacyBow = loadData("kmeans-spacy-bow")
    # _, nmiSpacyTfidf = loadData("kmeans-spacy-tfidf")
    # _, nmiSpacydoc2vec = loadData("kmeans-spacy-doc2vec")
    # _, nmiSpacyLdaBow = loadData("kmeans-spacy-lda-bow")
    # _, nmiSpacyLdaTfidf = loadData("kmeans-spacy-lda-tfidf")
    # plotNMI(nmiSpacyBow, nmiSpacyTfidf, nmiSpacydoc2vec, nmiSpacyLdaBow, nmiSpacyLdaTfidf, "Spacy")


    # Other training of D2Vec, shows similarity of topics to each other
    bow, tfidf, doc2VecFormat, id2word = loadData("embeddings-nltk")
    topicModel = trainDoc2VecByTopic(doc2VecFormat, save=True, name="nltk")
    similarTopics(topicModel.docvecs[0], topicModel, labels, 20, "nltk-topics")

    # Other number of clusters
    doc2vecRepNltk = loadData("doc2vec-rep-spacy")
    bestK(doc2vecRepNltk, labels)
    print("success")

if __name__ == "__main__":
    # execute only if run as a script
    main()
