from sklearn.manifold import TSNE
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import all_palettes
from gensim.models import LdaModel
from gensim import corpora
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import pyLDAvis.gensim
import pyLDAvis.sklearn
import numpy as np
import pandas as pd
    
from utils import load_data

# Takes ldamodel, id2word, and document representations (ie. bow, tfidf)

# Still would be nice: some sort of labeling for topics, to make t-SNE graph interactive
# This can be some separate array of strings where row number = doc, and value = doc's descriptor (maybe subject line?)
# We could then pass that in to the t-SNE graph as a hover
class LDAGrapher:
    
    def __init__(self, docRep, docRepName, id2word, ldamodel):
        self.docRep = docRep
        self.outPrefix = '../results/ldamodel-' + docRepName
        self.ldamodel = ldamodel
        self.corpus = Sparse2Corpus(docRep)#, document_columns=False)
        self.dictionary = Dictionary.from_corpus(self.corpus, id2word)

    
    def graphPyLDAvis(self):
        output_file = self.outPrefix + '-pyLDAvis.html';
        p = pyLDAvis.gensim.prepare(self.ldamodel, self.corpus, self.dictionary);
        pyLDAvis.save_html(p, output_file);


    def graphTSNE(self):
        output_file(self.outPrefix + '-tsne.html')
        
        # fit ldamodel so that it's size (num documents) x (num topics)
        ldamodel_fitted = []
        for i in range(len(self.corpus)):
            doc = [0] * 20
            for (x,y) in self.ldamodel[self.corpus[i]]:
                doc[x] = y
            ldamodel_fitted.append(doc)

        ldamodel_np = np.array(ldamodel_fitted)

        tsne = TSNE(perplexity=20, random_state=2017)
        tsne_embedding = tsne.fit_transform(ldamodel_np)
        tsne_embedding = pd.DataFrame(tsne_embedding, columns=['x','y'])
        tsne_embedding['hue'] = ldamodel_np.argmax(axis=1)

        source = ColumnDataSource(
            data=dict(
                x = tsne_embedding.x,
                y = tsne_embedding.y,
                colors = [all_palettes['Category20'][20][i] for i in tsne_embedding.hue],
                alpha = [0.9] * tsne_embedding.shape[0],
                size = [7] * tsne_embedding.shape[0]
            )
        )

        plot = figure(title="20 News Groups")
        plot.circle('x', 'y', size='size', fill_color='colors', 
                     alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="ldamodel_df")

        layout = column(plot)
        show(layout)

    def graphWordCloud(self):        
        topics = self.ldamodel.show_topics(num_topics=20, formatted=False)
    
        colors = all_palettes['Category20'][20]
        
        cloud = WordCloud(background_color='white',
                          max_words=20,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: colors[i],
                          prefer_horizontal=1.0)

        fig, axes = plt.subplots(4,5, figsize=(20,20), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i))
            plt.gca().axis('off')
    
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.savefig(self.outPrefix + '-wordcloud.png') 


def main():
    
    bow, tfidf, id2word = load_data('../results/preprocess.pkl')
    
    for (docRep, docRepName) in [(bow,'bow'), (tfidf,'tfidf')]:    
        ldamodel = LdaModel.load('../results/ldamodel-' + docRepName)
        grapher = LDAGrapher(docRep, docRepName, id2word, ldamodel) 
        
        print("Graphing t-SNE for " + docRepName + "...")
        grapher.graphTSNE()
        print("Graphing pyLDAvis for " + docRepName + "...")
        grapher.graphPyLDAvis()
        print("Creating word cloud for " + docRepName + "...")
        grapher.graphWordCloud()

    print("Done graphing!")


if __name__ == "__main__":
    # execute only if run as a script
    main()