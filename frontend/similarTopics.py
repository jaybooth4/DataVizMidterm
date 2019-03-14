import plotly.plotly as py
import plotly.offline as offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import numpy as np
from sklearn.decomposition import PCA
from .utils import loadData
from scipy import spatial

def similarTopics(topic, d2v_model, docLabels, name):
    testVec = d2v_model.docvecs[topic]
    docVecs = [(label, d2v_model.docvecs[label]) for label in set(docLabels)]
    # (Label, similarity)
    mostSim = list(map(lambda vec: (vec[0],1 - spatial.distance.cosine(testVec, vec[1])), docVecs))

    data = [go.Bar(
        x=list(map(lambda x: x[0], mostSim)),
        y=list(map(lambda x: x[1], mostSim))#simArr[:, 0],
        #simArr[:, 1]
    )]
    layout = go.Layout(
        title=('Most Similar Topics to ' + topic),
        xaxis=dict(
            title='Topics',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Percent Similar',
            ticklen=5,
            gridwidth=2,
        ),
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    offline.plot(fig, filename='results/D2VmostSimTopics' + name + '.html')
