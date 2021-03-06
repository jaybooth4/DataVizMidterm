import plotly.plotly as py
import plotly.offline as offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import numpy as np
from sklearn.decomposition import PCA
from .utils import loadData

def similarD2V(doc2Comp, d2v_model, d2vSize, name):
    # infer a vector of the new document
    newVec = d2v_model.infer_vector(doc2Comp)
    # find the most similar 10 documents
    mostSim = d2v_model.docvecs.most_similar([newVec], topn=10)
    # convert to numpy Arr
    simArr = np.array(mostSim)
    # make each number represent the percent similarity instead difference betweein 0-1
    for idx, each in enumerate(simArr):
        simArr[idx, 1] = 100-abs(float(simArr[idx, 1]))*100
    # graph it
    data = [go.Bar(
        x=simArr[:, 0],
        y=simArr[:, 1]
    )]
    layout = go.Layout(
        title=('Most Similar Documents to ' + "text 1"),
        xaxis=dict(
            title='Document Title',
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
    offline.plot(fig, filename='results/D2VmostSim' + name + '.html')
