from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
from sklearn.cluster import  MiniBatchKMeans

def bestK(data, labels):
    ''' Graph the sse vs clust count '''

    fileNameOut = "kmeans-analysis"

    costs = []
    K = range(1,25)
    for k in K:
        model = MiniBatchKMeans(n_clusters=k).fit(data)
        costs.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1)) / len(data))
    
    trace = go.Scatter(
        x=list(range(len(costs) + 1)),
        y=costs
    )

    layout = go.Layout(title="K Means Analysis")
    fig = go.Figure(data=[trace], layout=layout)
    offline.plot(fig, filename=fileNameOut + ".html")
