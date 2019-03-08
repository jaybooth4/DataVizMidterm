

def bagOfWords(rdd, tokenizer):
    return rdd.mapValues(lambda doc: tokenizer.tokenize(doc)).mapValues(lambda tokens: Counter(tokens))

def termFrequency(rdd):
    return rdd.flatMap(lambda (fileName, text): text.split()) \
        .map(lambda word: (toLowerCase(stripNonAlpha(word)), 1)) \
        .filter(lambda (word, count): word is not "") \
        .reduceByKey(add)


def documentFrequency(rdd, maxFreq, minFreq):
    fileCount = rdd.count()
    return rdd.flatMapValues(lambda text: text.split()) \
        .map(lambda (fileName, word): (fileName, toLowerCase(stripNonAlpha(word)))) \
        .distinct() \
        .filter(lambda (fileName, word): word != "") \
        .map(lambda (fileName, word): (word, 1.0)) \
        .reduceByKey(add) \
        .filter(lambda (word, count): count > min and count < max) \

def inverseDocumentFrequency(docFrequency, corpusCount):
    return docFrequency.map(lambda (word, count): (word, math.log(corpusCount / count)))

def TFIDF(termFreqRdd, idfRdd):
    return termFreqRdd.join(idfRdd) \
        .mapValues(lambda (tf, idf): tf * idf)

def getEmbeddings(data, labels):
    return None