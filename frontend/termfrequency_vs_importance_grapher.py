#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.offline as offline
import plotly.graph_objs as go

from grapher import Grapher
from utils import load_data

## Need term_frequency_vs_importance dataframe in this format:
# term_frequency_vs_importance_df = pd.DataFrame({
#     'term': random_words,
#     'term-frequency': random_term_frequency,
#     'importance': random_importance
# })
# with open('../backendOutput/term-frequency-vs-importance.pkl', 'wb') as f:
#     pickle.dump(term_frequency_vs_importance_df, f)

class TermFrequencyVsImportanceGrapher(Grapher):
    def graph(self, fileNameIn='term-frequency-vs-importance', fileNameOut='../results/term-frequency-vs-importance.html'):
        
        term_frequency_vs_importance_df = load_data(fileNameIn)

        trace = go.Scatter(
            x=term_frequency_vs_importance_df['term-frequency'], 
            y=term_frequency_vs_importance_df['importance'],
            mode='markers', 
            text=term_frequency_vs_importance_df['term'],
            name='term-frequency vs. importance')

        layout = go.Layout(title="Term-frequency vs. Importance",
                           xaxis={'title':'Importance'},
                           yaxis={'title':'Term-frequency'}
                          )

        fig = go.Figure(data=[trace], layout=layout)

        offline.plot(fig, filename=fileNameOut)
        
# How this should be graphed:
def main():
    grapher = TermFrequencyVsImportanceGrapher()
    grapher.graph()

if __name__ == "__main__":
    # execute only if run as a script
    main()
