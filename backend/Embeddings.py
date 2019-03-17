from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from .Util import saveData

# This file generates various embeddings of the dataset


def bagOfWordsFormat(data):
    ''' 
    Converts corpus into BOW format
    Input: List of lists of strings representing a list of tokens in a document 
    Output: CountVectorizer object, dict of id to word
    '''

    # Count vectorizer only accepts a list of strings, not a list of lists of tokens
    # The count vectorizer will split on . as a sentence terminator
    vectorizerData = list(map(lambda tokens: ".".join(tokens), data))
    countVectorizer = CountVectorizer()
    bow = countVectorizer.fit_transform(vectorizerData)
    id2word = dict((id, word)
                   for word, id in countVectorizer.vocabulary_.items())
    return bow, id2word


def TFIDFFormat(bowRep):
    ''' 
    Converts BOW into TFIDF format
    Input: CountVectorizer object 
    Output: TFIDFVectorizer object
    '''
    tfidf = TfidfTransformer()
    return tfidf.fit_transform(bowRep)


def doc2VecFormat(corpus, labels):
    ''' 
    Converts corpus into Doc2vec format 
    Input: list of list of string, list of labels
    Output: list of (list of strings, label)
    '''
    return [TaggedDocument(words=corpus[i], tags=[labels[i]]) for i in range(len(corpus))]


def getEmbeddings(data, labels, save=False, name=""):
    ''' 
    Generate all embeddings
    Input: list of list of string, list of labels, should file be saved?, name of file to save
    Output: BOW, TFIDF, Doc2vec, id2word objects 
    '''
    bow, id2word = bagOfWordsFormat(data)
    tfidf = TFIDFFormat(bow)
    doc2Vec = doc2VecFormat(data, labels)
    if save:
        saveData([bow, tfidf, doc2Vec, id2word], "embeddings-" + name)
    return bow, tfidf, doc2Vec, id2word
