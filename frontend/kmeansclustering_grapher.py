from gensim.matutils import Sparse2Corpus, corpus2dense
from sklearn.decomposition import PCA
from bokeh.models import ColumnDataSource
from bokeh.palettes import all_palettes
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.io import output_file

import numpy as np
import pandas as pd

from .utils import loadData

class KMeansClusteringGrapher:
    
    def __init__(self, num_docs, num_terms, dense_docs_rep, kmeans_labels):    
        self.num_docs = num_docs
        self.num_terms = num_terms
        self.docs_rep = dense_docs_rep
        self.kmeans_labels = kmeans_labels
    
    
    def graph(self, name):
        output_file("results/kmeans-clustering-" + name + ".html")
        
        pca = PCA(n_components=2)
        df = pd.DataFrame(pca.fit_transform(self.docs_rep), columns=['PCA1', 'PCA2'])

        source = ColumnDataSource(
            data = dict(
                PCA1 = df['PCA1'],
                PCA2 = df['PCA2'],
                colors = [all_palettes['Category20'][20][i] for i in self.kmeans_labels],
                alpha = [0.9] * self.num_docs,
                size = [7] * self.num_docs
            )
        )

        plot = figure(title="K-Means Clustering")
        plot.circle('PCA1', 
                    'PCA2', 
                    fill_color='colors',
                    alpha='alpha',
                    size='size',
                    source=source
                   )

        layout = column(plot)
        show(layout)
        
def convertToDenseRep(sparse, num_docs, num_terms):
    corpus = Sparse2Corpus(sparse, documents_columns=False)
    return np.matrix.transpose(corpus2dense(corpus, num_docs=num_docs, num_terms=num_terms))

def graphKMeansClusters(data, labels, isSparse, name):
    num_docs = num_terms = 0
    if isSparse:
        num_docs, num_terms = data.shape
        data = convertToDenseRep(data, num_docs, num_terms)
    else:
        num_docs = len(data)
        num_terms = len(data[0])

    grapher = KMeansClusteringGrapher(num_docs, num_terms, data, labels)         
    grapher.graph(name)