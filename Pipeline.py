from ml import *
from preprocess import *
import argparse

def main():    
    parser = argparse.ArgumentParser(description = 'Data Visualization Project Pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('data', default="100", help='Size of the data to read in') 
    parser.add_argument('preprocess', default="NLTK", help='Type of preprocessing to run on data', choices=['Spacy', 'NLTK', 'None'])
    parser.add_argument('save', default="NLTK", help='If results of pipeline should be saved')
    args = parser.parse_args()


    preprocessObj = preprocess.preprocess(args.preprocess)
    preprocessObj.preprocess(args.preprocess)

    # bow, tfidf, id2word = loadData("preprocess")
    # ldaModelBOW = trainLDA(bow, id2word)
    # ldaModelTFIDF = trainLDA(tfidf, id2word)
    # clusterData(bow, newsTrain.target[:dataSize])
    # clusterData(tfidf, newsTrain.target[:dataSize])


    # newsTrain = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    # newsTest = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    # preprocess = NLTKPreProcess() # SpacyPreprocess()
    # bow, tfidf, id2word = preprocess.preprocess(newsTrain.data[:dataSize], True)

if __name__ == "__main__":
    # execute only if run as a script
    main()
