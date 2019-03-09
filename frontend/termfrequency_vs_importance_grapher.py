#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.offline as offline
import plotly.graph_objs as go

from .utils import loadData

# Need term_frequency_vs_importance dataframe in this format:
# term_frequency_vs_importance_df = pd.DataFrame({
#     'term': random_words,
#     'term-frequency': random_term_frequency,
#     'importance': random_importance
# })


class TermFrequencyVsImportanceGrapher:

    def init(self, fileNameIn='backendOutput/stats.pkl'):
        _, _, tf, idf = loadData(fileNameIn)

        # Create dataframe
        term_frequency_df = pd.DataFrame(
            tf, columns=['term', 'term-frequency'])
        idf_frequency_df = pd.DataFrame(idf, columns=['term', 'importance'])
        self.term_frequency_vs_importance_df = pd.merge(
            term_frequency_df, idf_frequency_df, on='term', how='inner')

    def graph(self, fileNameOut='results/term-frequency-vs-importance.html'):

        trace = go.Scatter(
            x=self.term_frequency_vs_importance_df['term-frequency'],
            y=self.term_frequency_vs_importance_df['importance'],
            mode='markers',
            text=self.term_frequency_vs_importance_df['term'],
            name='term-frequency vs. importance')

        layout = go.Layout(title="Term-frequency vs. Importance",
                           xaxis={'title': 'Importance'},
                           yaxis={'title': 'Term-frequency'}
                           )

        fig = go.Figure(data=[trace], layout=layout)

        offline.plot(fig, filename=fileNameOut)


def graphTFVsImp(fileNameIn='backendOutput/stats.pkl', fileNameOut='results/term-frequency-vs-importance.html'):
    TFVIGrapher = TermFrequencyVsImportanceGrapher(fileNameIn)
    TFVIGrapher.graph(fileNameOut)
