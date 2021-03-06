import plotly.plotly as py
import plotly.offline as offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import numpy as np
from sklearn.decomposition import PCA
from .utils import loadData

def scatterD2V(d2vSize, d2v_model, name):
    #put vector representations into an array
    vecArray = np.zeros(shape=(len(d2v_model.docvecs), d2vSize))
    for idx in range(len(d2v_model.docvecs)):
        vecArray[idx]=d2v_model.docvecs[idx]
    #run through PCA
    d2vPca = PCA().fit_transform(vecArray)
    #visualize
    trace1 = go.Scatter(
        x = d2vPca[:, 0],
        y = d2vPca[:, 1],
        mode='markers',
        text = list(range(len(d2v_model.docvecs))),
        hoverinfo = 'text'
    )
    #identify the data
    data = [trace1]
    #make the layout
    layout = go.Layout(
        title=('Corpus Plotted in Term of Similarity'),
        xaxis= dict(
            title= 'PCA 2',
            ticklen= 5,
            zeroline= False,
            gridwidth= 2,
        ),
        yaxis=dict(
            title= 'PCA 1',
            ticklen= 5,
            gridwidth= 2,
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)

    offline.plot(fig, filename='results/D2VCorpusScatter' + name + '.html') #image = 'png',image_filename='D2VCorpusScatter',