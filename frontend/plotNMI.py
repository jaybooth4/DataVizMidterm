import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline


def plotNMI(nmiBow, nmiTfidf, nmidoc2vec, nmiLdaBow, nmiLdaTfidf, name):
    fileNameOut = "results/nmi-comparison" + name
    
    trace = go.Bar(
        x=["Bow", "TFIDF", "Doc2Vec", "LDABow", "LDATFIDF"],
        y=[nmiBow, nmiTfidf, nmidoc2vec, nmiLdaBow, nmiLdaTfidf]
    )

    layout = go.Layout(title="NMI Scores")
    fig = go.Figure(data=[trace], layout=layout)
    offline.plot(fig, filename=fileNameOut + ".html")