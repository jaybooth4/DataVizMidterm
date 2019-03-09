import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline


def boxPlot(names, sizes, fileNameOut="results/boxplot.py"):
    document_length_df = pd.DataFrame({
        'documentNum': names,
        'size': sizes,
    })

    trace = go.Box(
        name='Document Word Counts',
        y=document_length_df['size'],
        boxpoints='all'
    )

    layout = go.Layout(title="Document Size")
    fig = go.Figure(data=[trace], layout=layout)
    offline.plot(fig, filename=fileNameOut)
