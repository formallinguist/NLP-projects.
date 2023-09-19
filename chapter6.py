

#import numpy as np

#from nlpia.book.examples.ch06_nessvectors import *
#nessvector('Marie_Curie').round(2)

import numpy as np
import gensim
#print("NumPy version:", np.__version__)

from nlpia.data.loaders import get_data
word_vectors = get_data('word2vec')

print(word_vectors)


#from gensim.models.keyedvectors import keyedVectors
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format("/home/bluebird/nlpia/src/nlpia/bigdata/googlenews-vectors-negative300.bin.gz", binary=True)

print(word_vectors.most_similar(positive=['cooking','potatoes'], topn=5))

print(word_vectors.most_similar(positive=['germany','france'], topn=1))

print(word_vectors.doesnt_match("potatoes milk cake computer".split()))

print(word_vectors.most_similar(positive=['king','woman'], negative=['man'], topn=2))

print(word_vectors.similarity('princess','queen'))

print(word_vectors['phone'])


#Domain specific word to vec model.

with open('wiki.txt') as f:
    contents = f.read()
    #print(contents)



contents_len = len(contents)
print(contents_len)


import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

text = contents

sentences = sent_tokenize(text)
print(sentences)

words = []
for sentence in sentences:
    words.append(word_tokenize(sentence))
print(words)


from gensim.models.word2vec import Word2Vec

num_features = 300
min_word_count = 3
num_workers = 2
window_size = 6
subsampling = 1e-3

token_list = words

#Instantiating a word2vec model.

model = Word2Vec(
        token_list,
        workers = num_workers,
        vector_size = num_features,
        min_count = min_word_count,
        window = window_size,
        sample = subsampling)

model.init_sims(replace=True)

model_name = "my_domain_specific_word2vec_model"
model.save(model_name)

#Loading the saved model.

from gensim.models.word2vec import Word2Vec

model_name = "my_domain_specific_word2vec_model"
model = Word2Vec.load(model_name)
#model.most_similar('fruit')
print(model.wv.most_similar('Apple'))

#How to use pretrained FastText models


#import fasttext

#ft_model = fasttext.load_model('/mnt/c/Users/RAVI KIRAN/Downloads')

#ft_model.get_nearest_neighbors("potatoe")

#similar_words = ft_model.get_nearest_neighbors("soccer")

#print(similar_words)

import os

from nlpia.loaders import get_data 
from gensim.models.word2vec import KeyedVectors

wv = get_data('word2vec')
print(len(wv.index_to_key))

#Examine word2vec vocabulary frequencies.

import pandas as pd
vocab = pd.Series(wv.index_to_key)
print(vocab.iloc[1000000:1000006])
print(vocab.index[:10])  # Print the first 10 indices
print(vocab.index[-1])   # Print the last index

#Difference between  Illionos and Illini

import numpy as np
print(np.linalg.norm(wv['Illinois'] - wv['Illini']))

cos_similarity = np.dot(wv['Illinois'],wv['Illini']) / (
        np.linalg.norm(wv['Illinois']) *\
        np.linalg.norm(wv['Illini']))

print(cos_similarity)

print(1 - cos_similarity)


#Some Us city data.

from nlpia.data.loaders import get_data
cities = get_data('cities')
print(cities.head(1).T)

#Bubble chart of us cities.

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
us_300D = get_data('cities_us_wordvectors')
us_2d = pca.fit_transform(us_300D.iloc[:, :300])
print(us_2d)



