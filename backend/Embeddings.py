from collections import Counter
from .Util import parallelizeData
import uuid
import math
from operator import add
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def bagOfWordsFormat(data):
    # Count vectorizer only accepts a list of strings, not a list of lists of tokens
    vectorizerData = list(map(lambda tokens: " . ".join(tokens), data))
    # print(vectorizerData)
    countVectorizer = CountVectorizer(preprocessor=None, tokenizer=None, stop_words=None)
    bow = countVectorizer.fit_transform(vectorizerData)
    # print("Feature names")
    # print(countVectorizer.get_feature_names())
    print(countVectorizer.vocabulary_)
    id2word = dict((id, word)
                   for word, id in countVectorizer.vocabulary_.items())
    return bow, id2word


def TFIDFFormat(bowRep):
    tfidf = TfidfTransformer()
    return tfidf.fit_transform(bowRep)


def doc2VecFormat(corpus, labels):
    return [(corpus[i], labels[i]) for i in range(len(corpus))]


def getEmbeddings(data, labels):
    bow, id2word = bagOfWordsFormat(data)
    tfidf = TFIDFFormat(bow)
    doc2Vec = doc2VecFormat(data, labels)
    return bow, tfidf, doc2Vec, id2word
