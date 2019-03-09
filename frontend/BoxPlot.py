import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline


def boxPlot(names, sizes, title, fileNameOut="results/boxplot.html"):
    
    document_length_df = pd.DataFrame({
        'documentNum': names,
        'size': sizes,
    })

    trace = go.Box(
        name=title,
        y=document_length_df['size'],
        boxpoints='all'
    )

    layout = go.Layout(title=title, yaxis=dict(range=[0, 1000]))
    fig = go.Figure(data=[trace], layout=layout)
    offline.plot(fig, filename=fileNameOut + ".html")