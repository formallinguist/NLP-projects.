Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
#An example of splitting sentence into tokens.

sentence = "Thomas Jefferson began building Monticello at the age of 26."

sentence.split()
['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', '26.']

#Passing sentence into split function.

str.split(sentence)
['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', '26.']

#Creating numerical representation of each word as one-hot vectors.

import numpy as np
token_sequence = str.split(sentence)
vocab =  sorted(set(token_sequence))
', '.join(vocab)
'26., Jefferson, Monticello, Thomas, age, at, began, building, of, the'

num_tokens = len(token_sequence)
vocab_size = len(vocab)
onehot_vectors = np.zeros((num_tokens,vocab_size), int)

for i, word in enumerate(token_sequence):
    onehot_vectors[i, vocab.index(word)] = 1

    

' '.join(vocab)
'26. Jefferson Monticello Thomas age at began building of the'

onehot_vectors
array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


#One hot vectors sequence for monticello sentence.

import pandas as pd
pd.DataFrame(onehot_vectors, columns=vocab)
   26.  Jefferson  Monticello  Thomas  age  at  began  building  of  the
0    0          0           0       1    0   0      0         0   0    0
1    0          1           0       0    0   0      0         0   0    0
2    0          0           0       0    0   0      1         0   0    0
3    0          0           0       0    0   0      0         1   0    0
4    0          0           1       0    0   0      0         0   0    0
5    0          0           0       0    0   1      0         0   0    0
6    0          0           0       0    0   0      0         0   0    1
7    0          0           0       0    1   0      0         0   0    0
8    0          0           0       0    0   0      0         0   1    0
9    1          0           0       0    0   0      0         0   0    0
>>> 
>>> 
>>> #Cleaned one-hot vectors.
>>> 
>>> df = pd.DataFrame(onehot_vectors, columns=vocab)
>>> df[df == 0] =' '
>>> df
  26. Jefferson Monticello Thomas age at began building of the
0                               1                             
1             1                                               
2                                            1                
3                                                     1       
4                        1                                    
5                                      1                      
6                                                            1
7                                   1                         
8                                                        1    
9   1                                                         
>>> 
>>> 
>>> #Representation of the sentence in the binary bag of words.
>>> 
>>> sentence_bow = {}
>>> for token in sentence.split():
...     sentence_bow[token] = 1
... 
...     
>>> sorted(sentence_bow.items())
[('26.', 1), ('Jefferson', 1), ('Monticello', 1), ('Thomas', 1), ('age', 1), ('at', 1), ('began', 1), ('building', 1), ('of', 1), ('the', 1)]
>>> 
>>> 
>>> #sentence as binary bag of words vector.
>>> 
>>> sentence_bow = {}
for token in sentence.split():
    sentence_bow[token] = 1

    
sorted(sentence_bow.items())
[('26.', 1), ('Jefferson', 1), ('Monticello', 1), ('Thomas', 1), ('age', 1), ('at', 1), ('began', 1), ('building', 1), ('of', 1), ('the', 1)]


#Table of vectors corresponding to texts in corpus.

df = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence.split()])), columns=['sent']).T
df
      Thomas  Jefferson  began  building  Monticello  at  the  age  of  26.
sent       1          1      1         1           1   1    1    1   1    1


#Construct a dataFrame of Bag of words.

sentences = """Thomas Jefferson began building Monticello at the age of 26.\n"""
sentences += """ Construction was done mostly by local masons and carpenters.\n"""
sentences += "He moved into the south pavolion in 1770.\n"
sentences += """Turning Monticello into a neoclassical masterpiece was Jefferson's obsession.\n"""

corpus = {}

for i, sent in enumerate(sentence.split('\n')):
    corpus['sent{}'.format(i)] = dict((tok, 1) for tok in sent.split())

    

for i, sent in enumerate(sentences.split('\n')):
    corpus['sent{}'.format(i)] = dict((tok, 1) for tok in sent.split())

    

df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
df[df.columns[:10]]
       Thomas  Jefferson  began  building  Monticello  at  the  age  of  26.
sent0       1          1      1         1           1   1    1    1   1    1
sent1       0          0      0         0           0   0    0    0   0    0
sent2       0          0      0         0           0   0    1    0   0    0
sent3       0          0      0         0           1   0    0    0   0    0
sent4       0          0      0         0           0   0    0    0   0    0

#Dot product.

v1 = np.array([1, 2, 3])
v2 = np.array([2, 3, 4])

v1.dot(v2)
20

(v1 * v2).sum()
20

sum([x1 * x2 for x1, x2 in zip(v1, v2)])
20

#Measuring the overlap of wordcounts for the bag of words.

df = df.T
df.sent0.dot(df.sent1)
0
df.sent0.dot(df.sent2)
1
df.sent0.dot(df.sent3)
1


#To find the word that is shared between sent0 and sent3.

[(k, v) for (k, v) in (df.sent0 & df.sent3).items() if v]
[('Monticello', 1)]


#Tokenize the sentence with the regular expression.
import re
sentence = """Thomas Jefferson began building Monticello at the age of 26."""
tokens = re.split(r'[-\s.,;!?]+', sentence)
tokens
['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', '26', '']

#using re.compile to precompile te expression.

pattern = re.complie(r"([-\s.,;!?])+")
Traceback (most recent call last):
  File "<pyshell#121>", line 1, in <module>
    pattern = re.complie(r"([-\s.,;!?])+")
AttributeError: module 're' has no attribute 'complie'. Did you mean: 'compile'?
pattern = re.compile(r"([-\s.,;!?])+")
tokens = pattern.split(sentence)
tokens[-10:]
[' ', 'the', ' ', 'age', ' ', 'of', ' ', '26', '.', '']


sentence = """Thomas Jefferson began building Monticello  at the age of 26."""
tokens = pattern.split(sentence)
[x for x in tokens if x and x not in '- \t\n.,;!?']
['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', '26']



#Using RegexpTokenizer to tokenize the sentence.
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
tokenizer.tokenize(sentence)
['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', '26', '.']


#Using Treeword Tokenizer to tokenize the sentence.
from nltk.tokenize import TreebankWordTokenizer
sentence = """Monticello wasn't designated as UNESCO World Heritage Site until 1987."""
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize(sentence)
['Monticello', 'was', "n't", 'designated', 'as', 'UNESCO', 'World', 'Heritage', 'Site', 'until', '1987', '.']

#Tokenize the informal text from social networks such as twitter and facebook.

from nltk.tokenize.casual import casual_tokenize
message = """ RT @TJMonticello Best day everrrrrrrr at Monticello. Awessssssommmmmmeeeeee day :*)"""
casual_tokenize(message)
['RT', '@TJMonticello', 'Best', 'day', 'everrrrrrrr', 'at', 'Monticello', '.', 'Awessssssommmmmmeeeeee', 'day', ':*)']
casual_tokenize(message, reduce_len = True, strip_handles=True)
['RT', 'Best', 'day', 'everrr', 'at', 'Monticello', '.', 'Awesssommmeee', 'day', ':*)']


#1 gram tokenizer.
sentence = """Thomas Jefferson began building Monticello at the age of 26."""
pattern = re.compile(r"([-\s.,;!?])+")
tokens = pattern.split(sentence)
tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']
tokens
['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', '26']

#n-gram tokenizer using Nltk in action.

from nltk.util import ngrams
list(ngrams(tokens, 2))
[('Thomas', 'Jefferson'), ('Jefferson', 'began'), ('began', 'building'), ('building', 'Monticello'), ('Monticello', 'at'), ('at', 'the'), ('the', 'age'), ('age', 'of'), ('of', '26')]

list(ngrams(tokens, 3))
[('Thomas', 'Jefferson', 'began'), ('Jefferson', 'began', 'building'), ('began', 'building', 'Monticello'), ('building', 'Monticello', 'at'), ('Monticello', 'at', 'the'), ('at', 'the', 'age'), ('the', 'age', 'of'), ('age', 'of', '26')]


#Converting the above words that in tuple to strings.

two_grams = list(ngrams(tokens, 2))
[" ".join(x) for x in two_grams]
['Thomas Jefferson', 'Jefferson began', 'began building', 'building Monticello', 'Monticello at', 'at the', 'the age', 'age of', 'of 26']


#Stop words.

stop_words = ['a' , 'an' , 'the' , 'on' , 'of' , 'off' , 'this', 'is']
tokens = ['the' , 'house', 'is' , 'on' , 'fire']
tokens_without_stopwords = [x for x in tokens if x not in stop_words]
print(tokens_without_stopwords)
['house', 'fire']


#Nltk list of stopwords.

import nltk
nltk.download('stopwords')
[nltk_data] Downloading package stopwords to C:\Users\RAVI
[nltk_data]     KIRAN\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
True
stop_words = nltk.corpus.stopwords.words('english')
len(stop_words)
179

stop_words[:7]
['i', 'me', 'my', 'myself', 'we', 'our', 'ours']

[sw for sw in stopwords if len(sw) == 1]
Traceback (most recent call last):
  File "<pyshell#191>", line 1, in <module>
    [sw for sw in stopwords if len(sw) == 1]
NameError: name 'stopwords' is not defined. Did you mean: 'stop_words'?
[sw for sw in stop_words if len(sw) == 1]
['i', 'a', 's', 't', 'd', 'm', 'o', 'y']

#Nltk list of stopwords.

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
len(sklearn_stop_words)
318

len(stop_words)
179

set_sklearn_stop_words = set(sklearn_stop_words)
set_stop_words = set(stop_words)

len(set_stop_words.union(set_sklearn_stop_words))
378

len(set_stop_words.intersection(set_sklearn_stop_words))
119

#Noramlizing your vocabulary
# Case folding.

tokens =['House', 'Visitor' , 'Center']
normalized_tokens = [x.lower() for x in tokens]
print(normalized_tokens)
['house', 'visitor', 'center']


#Stemming:

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
' '.join([stemmer.stem(w).strip("'") for w in "dish washer's washed dishes".split()])
'dish washer wash dish'


#Lemmatization.

nltk.download('wordnet')
[nltk_data] Downloading package wordnet to C:\Users\RAVI
[nltk_data]     KIRAN\AppData\Roaming\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
True
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("better", pos = "a")
'good'

#VADER - A rule based sentiment analyzer.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()
sa.lexicon

[(tok, score) for tok, score in sa.lexicon.items()
 if " " in tok]
[("( '}{' )", 1.6), ("can't stand", -2.0), ('fed up', -1.8), ('screwed up', -1.5)]

sa.polarity_scores(text =\
                   "Python is very readable and it's great for NLP.")
{'neg': 0.0, 'neu': 0.661, 'pos': 0.339, 'compound': 0.6249}

sa.polarity_scores(text =\
                   "I am happy".)
SyntaxError: incomplete input
sa.polarity_scores(text =\
                   "I am happy.")
{'neg': 0.0, 'neu': 0.351, 'pos': 0.649, 'compound': 0.5719}
