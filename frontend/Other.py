import numpy as np
import pandas as pd

def boxPlot(names, sizes):
    document_length_df = pd.DataFrame({
        'document-name': random_document_name,
        'number-of-words': random_number_of_words,
    })

    document_length_df.head()

    trace = go.Box(
                name='Document Word Counts',
                y=document_length_df['number-of-words'],
                boxpoints='all'
    )

    data = [trace]

    iplot(data)