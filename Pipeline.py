from backend.Embeddings import getEmbeddings
from backend.Preprocess import preprocess
from backend.Util import createSparkContext, loadData
from backend.ML import clusterData
from sklearn.datasets import fetch_20newsgroups
from gensim.matutils import Sparse2Corpus
import argparse
import numpy as np


def main():
    # sc = createSparkContext()
    # corpus, labels = preprocess("NLTK", 100, sc, save=True)
    corpus, labels = loadData("preprocess")
    # bow, tfidf, doc2vecFormat, id2word = getEmbeddings(corpus, labels, save=True)
    bow, tfidf, doc2vecFormat, id2word = loadData("embeddings")
    

if __name__ == "__main__":
    # execute only if run as a script
    main()