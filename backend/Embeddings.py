from collections import Counter
import uuid
import math
from operator import add
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from .Util import saveData
from gensim.models.doc2vec import TaggedDocument


def bagOfWordsFormat(data):
    # Count vectorizer only accepts a list of strings, not a list of lists of tokens
    vectorizerData = list(map(lambda tokens: ".".join(tokens), data))
    countVectorizer = CountVectorizer()
    bow = countVectorizer.fit_transform(vectorizerData)
    id2word = dict((id, word)
                   for word, id in countVectorizer.vocabulary_.items())
    return bow, id2word


def TFIDFFormat(bowRep):
    tfidf = TfidfTransformer()
    return tfidf.fit_transform(bowRep)


def doc2VecFormat(corpus, labels):
    return [TaggedDocument(words=corpus[i], tags=[labels[i]]) for i in range(len(corpus))]


def getEmbeddings(data, labels, save=False):
    bow, id2word = bagOfWordsFormat(data)
    tfidf = TFIDFFormat(bow)
    doc2Vec = doc2VecFormat(data, labels)
    if save:
        saveData([bow, tfidf, doc2Vec, id2word], "embeddings")
    return bow, tfidf, doc2Vec, id2word
