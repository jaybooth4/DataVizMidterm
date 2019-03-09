from backend.Embeddings import getEmbeddings
from backend.Preprocess import preprocess
from backend.Util import createSparkContext, loadData
from backend.ML import trainLDA, trainDoc2Vec, clusterDataMiniBatch, getLDARep, getDoc2VecRep
from backend.Stats import getStats
from frontend.termfrequency_vs_importance_grapher import graphTFVsImp
from frontend.kmeansclustering_grapher import graphKMeansClusters
from frontend.BoxPlot import boxPlot
from sklearn.datasets import fetch_20newsgroups
from gensim.matutils import Sparse2Corpus
import argparse
import numpy as np
import time


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


    # Train models
    # bow, tfidf, doc2VecFormat, id2word = loadData("embeddings-nltk")
    # ldaModelBow = trainLDA(bow, id2word, save=True, name="nltk-bow")
    # ldaModelTfidf = trainLDA(tfidf, id2word, save=True, name="nltk-tfidf")
    # doc2VecModel = trainDoc2Vec(corpus, save=True, name="nltk")
    # ldaModelBow = loadData("ldamodel-nltk-bow")
    # ldaRepBow = getLDARep(ldaModelBow, bow, save=True, name="ldarep-nltk-bow")
    # ldaModelTfidf = loadData("ldamodel-nltk-tfidf")
    # ldaReptfidf = getLDARep(ldaModelTfidf, tfidf, save=True, name="ldarep-nltk-tfidf")
    # doc2VecModel = loadData("doc2vec-nltk")
    # doc2vecRep = getDoc2VecRep(doc2VecModel, save=True, name="doc2vec-rep-nltk")

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
    # doc2vecRepNltk = loadData("doc2vec-rep-nltk")
    # ldaRepNltkTfidf = loadData("ldarep-nltk-tfidf")
    # ldaRepNltkBow = loadData("ldarep-nltk-bow")
    # clusterDataMiniBatch(bow, labels, save=True, name="nltk-bow")
    # clusterDataMiniBatch(tfidf, labels, save=True, name="nltk-tfidf")
    # clusterDataMiniBatch(ldaRepNltkTfidf, labels, save=True, name="nltk-lda-tfidf")
    # clusterDataMiniBatch(ldaRepNltkBow, labels, save=True, name="nltk-lda-bow")
    # clusterDataMiniBatch(doc2vecRepNltk, labels, save=True, name="nltk-doc2vec")

    # doc2vecRepSpacy = loadData("doc2vec-rep-spacy")
    # ldaRepSpacyTfidf = loadData("ldarep-spacy-tfidf")
    # ldaRepSpacyBow = loadData("ldarep-spacy-bow")
    # clusterDataMiniBatch(bow, labels, save=True, name="spacy-bow")
    # clusterDataMiniBatch(tfidf, labels, save=True, name="spacy-tfidf")
    # clusterDataMiniBatch(ldaRepSpacyTfidf, labels, save=True, name="spacy-lda-tfidf")
    # clusterDataMiniBatch(ldaRepSpacyBow, labels, save=True, name="spacy-lda-bow")
    # clusterDataMiniBatch(doc2vecRepSpacy, labels, save=True, name="spacy-doc2vec")

    # Visualizations
    size, charSize, tf, idf = loadData("stats-nltk")

    # Box plot
    # boxPlot(range(len(size)), size, "Document Size", fileNameOut="results/boxplot-size")
    # boxPlot(range(len(size)), charSize, "Document Char Size", fileNameOut="results/boxplot-charsize")

    # TF vs Imp
    # graphTFVsImp(fileNameIn='backendOutput/stats-nltk.pkl', fileNameOut='results/term-frequency-vs-importance-nltk.html')
    # graphTFVsImp(fileNameIn='backendOutput/stats-spacy.pkl', fileNameOut='results/term-frequency-vs-importance-spacy.html')
    

    print("success")

if __name__ == "__main__":
    # execute only if run as a script
    main()