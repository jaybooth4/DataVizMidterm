This is a project to analyze and visualize 20 news groups data

## Analyses
Analyses include preprocessing/tokenization, creation of several embeddings (bag of words, tfidf), 
training of a doc2vec and lda model, and kmeans clustering to assess the accuracy in discriminating
topics. Please see the results folder for the results of the analyses.

## Pipeline
The project is coordinated by a pipeline file, which pulls in modules in the front/backend
folders to perform analysis and visualization. The results folder holds all final visualizations.

## Technologies
Apache Spark is used in the backend for the calculation of statistics and preprocessing. NLTK and Spacy
are used in the tokenization steps. Gensim and sklearn are used for nlp and clustering analysis steps.
Visualizations are made with Pandas, sklearn, Plotly, PyLDAViz, and bokeh.

## Modularity and Software Engineering Practices
Strong Software Engineering fundamentals were used in this project to ensure modularity, encapsulation,
and a clean user interface. A focus was on making sure that results could be saved using pickle to prevent
recalculation during each run. Also, apache spark was used in the backend step along with multicore 
training of models to ensure quick execution times. In the end a fairly general purpose library was created 
that users are able to understand quickly and interact with easily to do analyses and visualizaitions.