# set up fake data (remove when we have real data)
import urllib.request
import numpy as np

word_url = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
response = urllib.request.urlopen(word_url)
long_txt = response.read().decode()
words = long_txt.splitlines()

random_word_indices = np.random.randint(0,25486,100)
random_document_name = []
random_number_of_words = []
for index in random_word_indices:
    random_document_name.append(words[index])
    random_number_of_words.append(np.random.randint(1,500))

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