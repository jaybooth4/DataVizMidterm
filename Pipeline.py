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
    bow, tfidf, doc2vecFormat, id2word = getEmbeddings(corpus, labels)
    # for doc in corpus:
    #     print(doc)
    # print(type(bow.todense()))
    # print(bow.todense().shape)
    # np.apply_along_axis( lambda row: print(np.sum(row)), axis=1, arr=bow.todense() )
    # print(dir(bow.todense()))
    # for doc in bow.todense():
    #     print(doc)
    #     print([id2word[entry[0]] for entry in doc])
    # print(bow)
    # for doc in bow:
    #     print(doc)
    # print(dir(tfidf))


    # parser = argparse.ArgumentParser(description = 'Data Visualization Project Pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('data', default="100", help='Size of the data to read in')
    # parser.add_argument('preprocess', default="NLTK", help='Type of preprocessing to run on data', choices=['Spacy', 'NLTK', 'None'])
    # parser.add_argument('save', default="NLTK", help='If results of pipeline should be saved')
    # args = parser.parse_args()

    # bow, tfidf, id2word = loadData("preprocess")
    # ldaModelBOW = trainLDA(bow, id2word)
    # ldaModelTFIDF = trainLDA(tfidf, id2word)
    # clusterData(bow, newsTrain.target[:dataSize])
    # clusterData(tfidf, newsTrain.target[:dataSize])

    # newsTest = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    # preprocess = NLTKPreProcess() # SpacyPreprocess()
    # bow, tfidf, id2word = preprocess.preprocess(newsTrain.data[:dataSize], True)


if __name__ == "__main__":
    # execute only if run as a script
    main()