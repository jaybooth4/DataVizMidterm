from sklearn.manifold import TSNE
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column, gridplot
from bokeh.palettes import all_palettes
from gensim.models import LdaModel
from gensim import corpora
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from math import pi
import matplotlib.colors as mcolors
import pyLDAvis.gensim
import pyLDAvis.sklearn
import numpy as np
import pandas as pd
    
from .utils import loadData

# Takes doc_rep_name (ie. string "bow" or "tfidf", used to name output files), ldamodel, doc_rep as corpus (NOT sparse), dictionary (gensim Dictionary type), and document_labels

# Still would be nice: some sort of labeling for topics, to make t-SNE graph interactive
# This can be some separate array of strings where row number = doc, and value = doc's descriptor (maybe subject line?)
# We could then pass that in to the t-SNE graph as a hover
class LDAGrapher:
    
    def __init__(self, doc_rep_name, corpus, dictionary, ldamodel, document_labels):
        self.outPrefix = '../results/ldamodel-' + doc_rep_name
      
        self.ldamodel = ldamodel
        self.corpus = corpus
        self.dictionary = dictionary
        self.topics = self.ldamodel.show_topics(num_topics=20, formatted=False)
        self.document_labels = document_labels

    
    def graphPyLDAvis(self):
        output_file = self.outPrefix + '-pyLDAvis.html'
        p = pyLDAvis.gensim.prepare(self.ldamodel, self.corpus, self.dictionary)
        pyLDAvis.save_html(p, output_file)


    def graphTSNE(self, perplexity=20):
        ''' Runs t-SNE on document representations, then graphs groupings ''' 
        output_file(self.outPrefix + '-tsne.html')
        
        # fit ldamodel so that it's size (num documents) x (num topics)
        ldamodel_fitted = []
        for i in range(len(self.corpus)):
            doc = [0] * 20
            for (x,y) in self.ldamodel[self.corpus[i]]:
                doc[x] = y
            ldamodel_fitted.append(doc)

        ldamodel_np = np.array(ldamodel_fitted)

        tsne = TSNE(perplexity=20)
        tsne_embedding = tsne.fit_transform(ldamodel_np)
        tsne_embedding = pd.DataFrame(tsne_embedding, columns=['x','y'])
        tsne_embedding['hue'] = ldamodel_np.argmax(axis=1)

        source = ColumnDataSource(
            data=dict(
                x = tsne_embedding.x,
                y = tsne_embedding.y,
                doc_description = self.document_labels,
                colors = [all_palettes['Category20'][20][i] for i in tsne_embedding.hue],
                alpha = [0.9] * tsne_embedding.shape[0],
                size = [7] * tsne_embedding.shape[0]
            )
        )

        hover = HoverTool(
            tooltips = [
                ("Document", "@doc_description")
            ]
        )

        plot = figure(title="20 News Groups",
                      tools=[hover])
        plot.circle('x', 'y', size='size', fill_color='colors', 
                     alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="ldamodel_df")

        layout = column(plot)
        show(layout)


    def graphWordCloud(self):        
        colors = all_palettes['Category20'][20]
        cloud = WordCloud(background_color='white',
                          max_words=20,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: colors[i],
                          prefer_horizontal=1.0)

        fig, axes = plt.subplots(4,5, figsize=(20,20), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(self.topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i))
            plt.gca().axis('off')
    
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.savefig(self.outPrefix + '-wordcloud.png') 


    def graphWordWeight(self):
        output_file(self.outPrefix + '-word-weight.html')

        colors = [all_palettes['Category20'][20][i] for i in range(20)]
        plots = []
        for topic_number in range(20):
            topic_words = dict(self.topics[topic_number][1])
            topic_words_df = pd.DataFrame([[word, weight] for word, weight in topic_words.items()],
                                          columns=['word', 'weight'])

            plot = figure(x_range=list(topic_words_df['word'].unique()),
                          y_range=(0,1),
                          plot_width=250,
                          plot_height=250,
                         title="Topic " + str(topic_number))
            plot.vbar(x=topic_words_df['word'],
                      top=topic_words_df['weight'],
                      width=0.25,
                      color=colors[topic_number])
            plot.xaxis.major_label_orientation = pi/4

            plots.append(plot)

        grid = gridplot(plots, ncols=4)
        show(grid)
        

def graphLDA(embedFile='backendOutput/embeddings.pkl'):
    
    bow, tfidf, id2word = loadData(embedFile)
    
    for (docRep, docRepName) in [(bow,'bow'), (tfidf,'tfidf')]:
        ldamodel = LdaModel.load('backendOutput/ldamodel-' + docRepName + '.pkl')
        corpus = Sparse2Corpus(docRep, documents_columns=False)
        dictionary = Dictionary.from_corpus(corpus, id2word)
        #This could be more descriptive if we wanted
        document_labels = ["Document " + str(i) for i in range(100)]
        
        grapher = LDAGrapher(docRepName, corpus, dictionary, ldamodel, document_labels) 
        
        print("Graphing t-SNE for " + docRepName + "...")
        grapher.graphTSNE(perplexity=30)
        print("Graphing pyLDAvis for " + docRepName + "...")
        grapher.graphPyLDAvis()
        print("Creating word cloud for " + docRepName + "...")
        grapher.graphWordCloud()
        print("Graphing word weights for " + docRepName + "...")
        grapher.graphWordWeight()

    print("Done graphing!")