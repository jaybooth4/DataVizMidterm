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
    
    
    def graph(self, outFile='results/kmeans-clustering.html'):
        output_file(outFile)
        
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
        
def convertToDenseRep(sparse, num_terms, num_docs):
    corpus = Sparse2Corpus(sparse, documents_columns=False)
    return np.matrix.transpose(corpus2dense(corpus, num_terms=num_terms, num_docs=num_docs))

def graphKMeansClusters(data, labels, isSparse, num_terms = 5000, num_docs = 100):
    
    if isSparse:
        data = convertToDenseRep(data, num_terms, num_docs)

    grapher = KMeansClusteringGrapher(num_docs, num_terms, data, labels)         
    grapher.graph()