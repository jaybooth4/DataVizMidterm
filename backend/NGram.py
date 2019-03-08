#!/usr/bin/env python
# coding: utf-8

from gensim.models.phrases import Phrases

def convertNGram(corpus):
    ngram = Phrases(corpus)
    return [ngram[doc] for doc in corpus]