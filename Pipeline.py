from backend.Embeddings import getEmbeddings
from backend.Preprocess import preprocess
from backend.Util import createSparkContext, loadData
from sklearn.datasets import fetch_20newsgroups
import argparse


def main():
    sc = createSparkContext()
    corpus, labels = preprocess("NLTK", 100, sc)#, save=True)
    # corpus, labels = loadData("preprocess")
    # bow, tfidf, doc2vecFormat = getEmbeddings(corpus, labels, sc)

    print(corpus[:20])
    print(fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), download_if_missing=True).data[:20])


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
