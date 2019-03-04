#!/usr/bin/env python
# coding: utf-8

from abc import ABCMeta, abstractmethod

class Grapher(metaclass=ABCMeta):

    @abstractmethod
    def graph(self, fileNameIn, fileNameOut):
        pass
