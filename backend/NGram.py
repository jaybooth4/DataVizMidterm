#!/usr/bin/env python
# coding: utf-8

from gensim.models.phrases import Phrases


def convertNGram(corpus):
    '''
    Generate ngrams for dataset
    input: list of list of tokens
    output: list of list of tokens with ngrams included
    '''
    ngram = Phrases(corpus)
    return [ngram[doc] for doc in corpus]
